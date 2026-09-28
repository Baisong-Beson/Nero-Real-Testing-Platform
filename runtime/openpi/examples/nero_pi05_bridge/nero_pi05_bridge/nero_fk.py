"""Optional pinocchio FK helpers for Gate 2 TCP jump / workspace checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

GRIPPER_MIN_M = 0.030
GRIPPER_MAX_M = 0.099
TCP_FRAME = "gripper_base"


class NeroFK:
    """Thin pinocchio wrapper; fails closed if pinocchio/URDF unavailable."""

    def __init__(self, urdf: Path, *, tcp_frame: str = TCP_FRAME):
        try:
            import pinocchio as pin
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "pinocchio is required for Gate 2 TCP checks; run under conda env pika "
                "or install pinocchio in the ROS Python environment"
            ) from exc
        self._pin = pin
        self.model = pin.buildModelFromUrdf(str(Path(urdf).expanduser().resolve()))
        self.data = self.model.createData()
        if not self.model.existFrame(tcp_frame):
            raise ValueError(f"URDF missing TCP frame {tcp_frame}")
        self.frame_id = self.model.getFrameId(tcp_frame)
        self.tcp_frame = tcp_frame

    def _configuration(self, joints7: np.ndarray, gripper_width_m: float) -> np.ndarray:
        width = float(np.clip(gripper_width_m, 0.0, 0.100))
        return np.concatenate(
            (np.asarray(joints7, dtype=np.float64), [width, width / 2.0, -width / 2.0])
        )

    def tcp_xyz(self, joints7: np.ndarray, gripper_width_m: float = GRIPPER_MAX_M) -> np.ndarray:
        q = self._configuration(joints7, gripper_width_m)
        self._pin.forwardKinematics(self.model, self.data, q)
        self._pin.updateFramePlacements(self.model, self.data)
        return np.asarray(self.data.oMf[self.frame_id].translation.copy(), dtype=np.float64)

    def tcp_from_closedness(self, joints7: np.ndarray, closedness: float) -> np.ndarray:
        closed = float(np.clip(closedness, 0.0, 1.0))
        width = GRIPPER_MAX_M - closed * (GRIPPER_MAX_M - GRIPPER_MIN_M)
        return self.tcp_xyz(joints7, width)
