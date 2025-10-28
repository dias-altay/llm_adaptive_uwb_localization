import asyncio, json, time, threading
from typing import Callable, Optional, Dict, Any

try:
    import websockets  # type: ignore
except Exception as _e:
    websockets = None

def _norm_obj_row(ts: float, frame: int, obj_name: str, obj: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "timestamp": float(ts),
        "frame": int(frame),
        "mode": 1,
        "object": str(obj_name),
        "id": obj.get("id"),
        "x": obj.get("x"),
        "y": obj.get("y"),
        "z": obj.get("z"),
        "qx": obj.get("qx"),
        "qy": obj.get("qy"),
        "qz": obj.get("qz"),
        "qw": obj.get("qw"),
    }

async def _ws_loop(url: str, on_row: Callable[[Dict[str, Any]], None], stop_flag: threading.Event,
                   object_filter: Optional[str], id_filter: Optional[str]):
    if websockets is None:
        raise RuntimeError("websockets package not installed")
    frame_counter = 0
    while not stop_flag.is_set():
        try:
            async with websockets.connect(url) as ws:
                while not stop_flag.is_set():
                    msg = await ws.recv()
                    now = time.time()
                    try:
                        data = json.loads(msg)
                    except Exception:
                        continue
                    ts = float(data.get("timestamp", now))
                    frame = int(data.get("frame", 0) or 0)
                    if frame <= 0:
                        frame_counter += 1
                        frame = frame_counter
                    objs = data.get("objects", {})
                    for name, obj in objs.items():
                        if object_filter is not None and name != object_filter:
                            continue
                        if id_filter is not None and str(obj.get("id")) != str(id_filter):
                            continue
                        on_row(_norm_obj_row(ts, frame, name, obj))
        except Exception:
            await asyncio.sleep(0.5)

def start_ws_receiver(url: str, on_row: Callable[[Dict[str, Any]], None],
                      object_filter: Optional[str] = None, id_filter: Optional[str] = None):
    """
    Start a background WebSocket receiver. Returns (thread, stop_event).
    """
    stop_flag = threading.Event()
    def _runner():
        asyncio.run(_ws_loop(url, on_row, stop_flag, object_filter, id_filter))
    th = threading.Thread(target=_runner, daemon=True)
    th.start()
    return th, stop_flag
