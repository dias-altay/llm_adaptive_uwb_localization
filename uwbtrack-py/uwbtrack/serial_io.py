import time, re
import serial

re_meas = re.compile(r"MEAS\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([0-9.]+)\s*,\s*(-?[0-9.]+)")

def init_serial_connection(port: str, baud: int = 115200):
    ser = serial.Serial(port, baudrate=baud, timeout=0.2)
    ser.write(b"S"); ser.flush()
    time.sleep(0.2)
    return ser

def serial_lines(ser):
    buf = bytearray()
    try:
        while True:
            data = ser.read(4096)
            if data:
                buf.extend(data)
                while True:
                    nl = buf.find(b"\n")
                    if nl < 0: break
                    line = buf[:nl]; del buf[:nl+1]
                    try:
                        yield line.decode(errors="ignore").strip()
                    except:
                        continue
            else:
                time.sleep(0.01)
    except GeneratorExit:
        raise

def parse_meas(line: str):
    if "MEAS" not in line:
        return None
    parts = [p.strip() for p in line.split(",")]
    try:
        seq = int(parts[1]); aid = int(parts[2])
        rng = float(parts[3].replace("..", ".")); ci = float(parts[4].replace("..", "."))
    except Exception:
        return None
    m = {"seq": seq, "aid": aid, "range": rng, "ci": ci}
    try:
        fp1 = int(parts[9]); fp2 = int(parts[10]); fp3 = int(parts[11])
        maxNoise = int(parts[12]); stdNoise = int(parts[13]); rpc = int(parts[14])
        m.update({"fp1": fp1, "fp2": fp2, "fp3": fp3,
                  "maxNoise": maxNoise, "stdNoise": stdNoise, "rpc": rpc})
    except Exception:
        pass
    return m
