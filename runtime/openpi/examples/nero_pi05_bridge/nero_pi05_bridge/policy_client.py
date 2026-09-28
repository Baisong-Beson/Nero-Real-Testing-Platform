"""Async WebSocket client for the OpenPI policy server."""

from __future__ import annotations

import asyncio

import websockets

from nero_pi05_bridge import msgpack_numpy


class PolicyClient:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        api_key: str = "",
        connect_timeout_sec: float = 5.0,
        inference_timeout_sec: float = 30.0,
    ) -> None:
        self._uri = host if host.startswith(("ws://", "wss://")) else f"ws://{host}:{port}"
        self._api_key = api_key
        self._connect_timeout_sec = connect_timeout_sec
        self._inference_timeout_sec = inference_timeout_sec
        self._websocket = None
        self.metadata = {}

    @property
    def uri(self) -> str:
        return self._uri

    async def connect(self) -> dict:
        await self.close()
        # websockets>=13 renamed extra_headers -> additional_headers; openpi-client
        # already uses the new name. Passing the old kwarg into 15/16 forwards it to
        # asyncio.create_connection and crashes with TypeError.
        kwargs: dict = {
            "compression": None,
            "max_size": None,
        }
        if self._api_key:
            kwargs["additional_headers"] = {
                "Authorization": f"Api-Key {self._api_key}"
            }
        connect = websockets.connect(self._uri, **kwargs)
        self._websocket = await asyncio.wait_for(connect, timeout=self._connect_timeout_sec)
        payload = await asyncio.wait_for(self._websocket.recv(), timeout=self._connect_timeout_sec)
        if isinstance(payload, str):
            raise RuntimeError(f"OpenPI server returned text metadata: {payload}")
        self.metadata = msgpack_numpy.unpackb(payload)
        return self.metadata

    async def infer(self, observation: dict) -> dict:
        if self._websocket is None:
            raise RuntimeError("Policy client is not connected")
        await asyncio.wait_for(
            self._websocket.send(msgpack_numpy.packb(observation)),
            timeout=self._inference_timeout_sec,
        )
        payload = await asyncio.wait_for(
            self._websocket.recv(),
            timeout=self._inference_timeout_sec,
        )
        if isinstance(payload, str):
            raise RuntimeError(f"OpenPI inference error:\n{payload}")
        return msgpack_numpy.unpackb(payload)

    async def close(self) -> None:
        if self._websocket is not None:
            websocket = self._websocket
            self._websocket = None
            await websocket.close()
