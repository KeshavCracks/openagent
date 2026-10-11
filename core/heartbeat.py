"""Background keep-alive heartbeat.

Google Colab / Kaggle idle-reaps VMs while the doctor reviews charts.
This module runs an asyncio background task that pings the remote gateway
health endpoint through the tunnel every ~3 minutes (± jitter), using the
same Cloudflare-bypass headers and bearer token as clinical traffic.

Failure semantics: three consecutive failures mark the tunnel unhealthy
(surfaced in the status bar) but the monitor never dies — transient
tunnel outages self-heal without user action.
"""
from __future__ import annotations

import asyncio
import random
import time
from typing import Optional

from core.tunnel_client import TunnelClient, TunnelError


class HeartbeatMonitor:
    def __init__(
        self,
        tunnel: TunnelClient,
        interval_s: float = 180.0,
        jitter_s: float = 15.0,
        failure_threshold: int = 3,
    ) -> None:
        self.tunnel = tunnel
        self.interval_s = interval_s
        self.jitter_s = jitter_s
        self.failure_threshold = failure_threshold
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        # observability
        self.pings_sent = 0
        self.consecutive_failures = 0
        self.last_ok_at: Optional[float] = None
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------- state --
    @property
    def healthy(self) -> bool:
        return self.last_ok_at is not None and self.consecutive_failures < self.failure_threshold

    def status(self) -> dict:
        return {
            "running": self._task is not None and not self._task.done(),
            "healthy": self.healthy,
            "pings_sent": self.pings_sent,
            "consecutive_failures": self.consecutive_failures,
            "last_ok_at": self.last_ok_at,
            "last_error": self.last_error,
        }

    # -------------------------------------------------------- lifecycle ---
    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.get_running_loop().create_task(self._loop(), name="medharness-heartbeat")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    # -------------------------------------------------------------- loop --
    async def _loop(self) -> None:
        first = True
        while not self._stop.is_set():
            if not first:  # ping immediately on start, then on interval
                interval = self.interval_s + random.uniform(0.0, self.jitter_s)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=interval)
                    break  # stop event fired
                except asyncio.TimeoutError:
                    pass
            first = False
            self.pings_sent += 1
            try:
                await self.tunnel.health()
                self.consecutive_failures = 0
                self.last_ok_at = time.time()
                self.last_error = None
            except TunnelError as exc:
                self.consecutive_failures += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
