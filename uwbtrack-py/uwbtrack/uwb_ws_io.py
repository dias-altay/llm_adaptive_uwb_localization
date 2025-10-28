import asyncio
import threading
import time
from queue import Queue, Empty
from typing import Optional

import websockets


class UWBWebSocketDevice:
    """
    Serial-like wrapper around a UWB WebSocket text stream.
    - read(n): returns bytes (with '\n' as line separators) similar to serial.Serial.read()
    - write/flush/reset_input_buffer: no-ops (kept for compatibility)
    - close(): stop background reader and close connection
    """
    def __init__(self, url: str, timeout: float = 0.2, retry_start: float = 1.0, retry_max: float = 5.0):
        self.url = url
        self.timeout = timeout
        self.retry_start = retry_start
        self.retry_max = retry_max

        self._stop = threading.Event()
        self._queue: Queue[bytes] = Queue(maxsize=4096)
        self._buf = bytearray()
        self._thread = threading.Thread(target=self._run_loop, name="UWBWSReader", daemon=True)
        self._thread.start()

    def read(self, n: int) -> bytes:
        """
        Return up to n bytes if available; otherwise block up to self.timeout,
        returning b"" on timeout (matching serial.Serial(timeout=...)).
        """
        if n <= 0:
            return b""
        if not self._buf:
            try:
                chunk = self._queue.get(timeout=self.timeout)
                if chunk:
                    self._buf.extend(chunk)
            except Empty:
                return b""
        if not self._buf:
            return b""
        out = self._buf[:n]
        del self._buf[:n]
        return bytes(out)

    def write(self, _data: bytes) -> int:
        return len(_data) if _data is not None else 0

    def flush(self):
        pass

    def reset_input_buffer(self):
        self._buf.clear()
        try:
            while True:
                self._queue.get_nowait()
        except Empty:
            pass

    def close(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run_loop(self):
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._read_forever())
        finally:
            try:
                loop.stop()
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass

    async def _read_forever(self):
        backoff = self.retry_start
        while not self._stop.is_set():
            try:
                async with websockets.connect(self.url) as ws:
                    backoff = self.retry_start
                    while not self._stop.is_set():
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue
                        except Exception:
                            break
                        if msg is None:
                            break
                        if isinstance(msg, bytes):
                            data = msg
                        else:
                            data = msg.encode("utf-8", errors="ignore")
                        if not data.endswith(b"\n"):
                            data += b"\n"
                        try:
                            self._queue.put_nowait(data)
                        except Exception:
                            try:
                                _ = self._queue.get_nowait()
                            except Empty:
                                pass
                            try:
                                self._queue.put_nowait(data)
                            except Exception:
                                pass
            except Exception:
                await asyncio.sleep(backoff)
                backoff = min(self.retry_max, backoff * 1.7)

def init_ws_connection(url: str, timeout: float = 0.2) -> UWBWebSocketDevice:
    """
    Create a serial-like device backed by a UWB WebSocket stream.
    """
    return UWBWebSocketDevice(url=url, timeout=timeout)
