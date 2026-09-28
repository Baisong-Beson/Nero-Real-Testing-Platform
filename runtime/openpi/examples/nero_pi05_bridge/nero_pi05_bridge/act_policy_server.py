"""ACT inference server for the NERO real-robot round-1 export.

The websocket shape matches ``PolicyClient`` so the existing ROS tooling can
be reused.  This server is inference-only and advertises ``commandable=false``;
it has no ROS imports and never publishes a control message.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.machinery
import importlib.util
import json
import hashlib
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch



ROOT = Path(__file__).resolve().parents[3]
EXPORT = ROOT.parents[1] / "models/act_export"
MPI_EVAL = ROOT.parent / "mpi_eval"
SNAPSHOT = ROOT / "artifacts/nero_real_round1_20260921/a100_snapshot"
ACT_ROOT = SNAPSHOT / "refs/act/robotwin_act_20260921/cobodied_eval"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _act_head():
    package = "act_runtime"
    sys.modules.setdefault(package, importlib.util.module_from_spec(importlib.machinery.ModuleSpec(package, None)))
    trans = _load_module(package + ".act_transformer", ACT_ROOT / "act_transformer.py")
    trans.__package__ = package
    head = _load_module(package + ".act_head", ACT_ROOT / "act_head.py")
    head.__package__ = package
    return head.ACTHead


def _letterbox(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    out = np.zeros((size, size, 3), dtype=np.uint8)
    x, y = (size - nw) // 2, (size - nh) // 2
    out[y : y + nh, x : x + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def module_sha256(module):
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class VisualEncoder:
    def __init__(self, name: str, device: torch.device):
        self.name = name
        self.device = device
        import timm  # Required real dependency, never a stub.
        legacy = _load_module("act_legacy_encoders", SNAPSHOT / "round1/legacy_encoders.py")
        legacy.WORKSPACE = MPI_EVAL
        legacy.CHECKPOINTS["p0"] = (
            MPI_EVAL / "checkpoints/r14-handoff-20260910/p0_step6000/model.ckpt",
            "d7c139a11217468d6bd2f712df92f44bf07b2e82c1164b49b35c776138385f12",
        )
        if name in ("mpi-base", "p0"):
            self.model = legacy.FrozenEncoder(name, device=str(device))
            if name == "mpi-base":
                # Keep the raw MPI backbone because its public
                # get_representations() path inserts a CLS token and does not
                # reproduce the round-1 duplicated-view token layout.
                self.mpi_backbone = self.model.model
                self.mpi_transform = self.model.transform
                for key in list(self.mpi_backbone._modules):
                    if key not in {"patch2embed", "encoder_blocks", "encoder_norm"}:
                        delattr(self.mpi_backbone, key)
                for key in list(self.mpi_backbone._parameters):
                    if key not in {"img_token", "encoder_pe", "ctx_enc_pe"}:
                        delattr(self.mpi_backbone, key)
            return
        if name != "g2":
            raise ValueError(name)
        vit_path = Path(os.environ.get("NERO_ACT_VIT_PATH", ""))
        if not vit_path.is_file():
            raise RuntimeError("Set NERO_ACT_VIT_PATH to the external encoder source before using this model")
        vit = _load_module("act_g2_vit", vit_path)
        self.model = vit.ViT(img_size=(448, 448), patch_size=16, embed_dim=1280, depth=32,
                             num_heads=16, mlp_ratio=4, qkv_bias=True, drop_path_rate=0,
                             use_checkpoint=False)
        scene = MPI_EVAL / "checkpoints/EgoHEP_G2_s42_step2000_phi2/scene.ckpt"
        if file_sha256(scene) != "dea36156c97ff85de15f9f570b5e2d9ff856732df2cc223d32996edaddca10bc":
            raise ValueError("G2 scene checkpoint hash mismatch")
        payload = torch.load(scene, map_location="cpu", weights_only=True)
        selected = {k[len("scene_encoder.model.") :]: v for k, v in payload["state_dict"].items() if k.startswith("scene_encoder.model.")}
        if len(selected) != 389:
            raise RuntimeError(f"G2 scene tensor count {len(selected)} != 389")
        self.model.load_state_dict(selected, strict=True)
        self.model.to(device=device, dtype=torch.float32).requires_grad_(False).eval()

    @torch.inference_mode()
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        if self.name == "mpi-base":
            from PIL import Image
            x = torch.stack([self.mpi_transform(Image.fromarray(image))]).to(self.device)
            b = len(x)
            m = self.mpi_backbone
            patches = m.patch2embed(torch.stack((x, x), 1).flatten(0, 1)) + m.encoder_pe
            patches = patches.reshape(b, 2, 196, 768) + m.ctx_enc_pe[:, 0:1]
            tokens = patches.transpose(1, 2).reshape(b, 392, 768) + m.img_token
            mask = torch.ones(b, 392, device=self.device, dtype=torch.long)
            for block in m.encoder_blocks:
                tokens = block(tokens, mask)
            return m.encoder_norm(tokens).reshape(b, 196, 2, 768).mean(2).contiguous().float()
        if self.name == "p0":
            raw = self.model([image])
            raw = raw.transpose(1, 2).reshape(1, 1280, 28, 28)
            return torch.nn.functional.avg_pool2d(raw, 2).flatten(2).transpose(1, 2).contiguous().float()
        size = _letterbox(np.asarray(image, dtype=np.uint8), 448)
        mean = np.array([.485, .456, .406], np.float32)
        std = np.array([.229, .224, .225], np.float32)
        x = torch.from_numpy(((size.astype(np.float32) / 255 - mean) / std).transpose(2, 0, 1).copy())[None].to(self.device)
        raw = self.model(x).flatten(2).transpose(1, 2)
        return torch.nn.functional.avg_pool2d(raw.transpose(1, 2).reshape(1, 1280, 28, 28), 2).flatten(2).transpose(1, 2).contiguous().float()


class ACTPolicy:
    def __init__(self, path: Path, device: torch.device):
        self.path = path.resolve()
        path = self.path
        self.device = device
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        export = path.parents[2]
        seal = json.loads((export / "EXPORT_COMPLETE").read_text())
        if file_sha256(export / "manifest.json") != seal["manifest_sha256"]:
            raise ValueError("export seal mismatch")
        manifest = json.loads((export / "manifest.json").read_text())
        identity = file_sha256(path)
        if identity != manifest["files"][str(path.relative_to(export))]["sha256"]:
            raise ValueError("policy hash mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema") != "real_robot_round1.inference_policy.v1":
            raise ValueError(f"unsupported policy schema: {payload.get('schema')}")
        if payload["config"].get("state_dim") != 16 or payload["config"].get("action_dim") != 16:
            raise ValueError("ACT policy must be bimanual 16D")
        ACTHead = _act_head()
        self.head = ACTHead(payload["config"], state_dim=16, action_dim=16)
        self.head.load_state_dict(payload["head"], strict=True)
        self.head.to(device=device, dtype=torch.float32).eval()
        self.norm = {k: torch.as_tensor(v, device=device, dtype=torch.float32) for k, v in payload["normalization"].items()}
        if module_sha256(self.head) != payload["head_sha256"]:
            raise ValueError("head tensor hash mismatch")
        self.encoder = VisualEncoder(payload["model"], device)
        wrapper = self.encoder.model
        if payload["model"] == "g2":
            wrapper = torch.nn.Module()
            wrapper.add_module("model", self.encoder.model)
        encoder_hash = module_sha256(wrapper)
        if encoder_hash != payload["encoder"]["encoder_state_sha256"]:
            raise ValueError("loaded encoder tensor hash mismatch")
        if any(p.requires_grad for p in wrapper.parameters()) or wrapper.training and payload["model"] != "g2":
            raise ValueError("encoder must be frozen/eval")
        self.metadata = {"schema": payload["schema"], "task": payload["task"], "model": payload["model"],
                         "commandable": False, "policy_sha256": identity, "encoder_state_sha256": encoder_hash, "action_shape": [16, 16], "latent": "zero",
                         "preprocessing": payload["config"]["preprocessing_protocol"]}

    @torch.inference_mode()
    def infer(self, observation: dict) -> dict:
        image = np.asarray(observation["observation/exterior_image_1_left"])
        if image.dtype != np.uint8:
            raise ValueError("RGB must be uint8")
        state = np.array(observation["observation/state"], dtype=np.float32, copy=True)
        if image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) <= 0 or state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError(f"bad ACT observation image={image.shape} state={state.shape}")
        started = time.monotonic()
        features = self.encoder(image)
        normalized = (torch.from_numpy(state).to(self.device) - self.norm["state_mean"]) / self.norm["state_std"]
        pred, mu, logvar = self.head(features, normalized[None])
        if mu is not None or logvar is not None:
            raise RuntimeError("inference unexpectedly sampled ACT posterior")
        actions = pred[0] * self.norm["action_std"] + self.norm["action_mean"]
        if actions.shape != (16, 16) or not torch.isfinite(actions).all():
            raise RuntimeError("ACT output contains NaN/Inf")
        return {"actions": actions.cpu().numpy().astype(np.float32),
                "server_timing": {"infer_ms": (time.monotonic() - started) * 1000}}


async def _serve(policy: ACTPolicy, host: str, port: int):
    import websockets.asyncio.server as ws_server
    from nero_pi05_bridge import msgpack_numpy
    packer = msgpack_numpy.Packer() if hasattr(msgpack_numpy, "Packer") else None
    pack = packer.pack if packer else msgpack_numpy.packb

    async def handler(websocket):
        await websocket.send(pack(policy.metadata))
        async for payload in websocket:
            obs = msgpack_numpy.unpackb(payload)
            await websocket.send(pack(policy.infer(obs)))

    async with ws_server.serve(handler, host, port, compression=None, max_size=None):
        logging.info("ACT server listening on %s:%d", host, port)
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("banana", "holder", "plate"), required=True)
    parser.add_argument("--model", choices=("mpi-base", "p0", "g2"), required=True)
    parser.add_argument("--export", type=Path, default=EXPORT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8016)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    policy = ACTPolicy(args.export / args.task / args.model / "policy.pt", device)
    asyncio.run(_serve(policy, args.host, args.port))


if __name__ == "__main__":
    main()
