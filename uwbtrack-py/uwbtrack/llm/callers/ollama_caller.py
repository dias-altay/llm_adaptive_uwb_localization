import requests, json, time
from typing import List, Dict, Any
from urllib.parse import urlparse

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "mode": {"type": "string", "enum": ["static", "moving", "erratic"]},
        "kf": {
            "type": "object",
            "properties": {
                "process_var": {
                    "type": "number",
                    "minimum": 0.03,
                    "maximum": 1.50,
                    "description": "Must be within mode-specific ranges: static=[0.03,0.05], moving=[0.30,1.00], erratic=[0.80,1.50]"
                },
                "meas_var": {
                    "type": "number",
                    "minimum": 0.03,
                    "maximum": 0.25,
                    "description": "Must be within mode-specific ranges: static=[0.10,0.18], moving=[0.04,0.12], erratic=[0.12,0.25]"
                },
                "innovation_max": {
                    "type": "number",
                    "minimum": 0.60,
                    "maximum": 2.00,
                    "description": "Must be within mode-specific ranges: static=[0.70,0.90], moving=[1.40,1.90], erratic=[1.80,2.00]"
                }
            },
            "required": ["process_var", "meas_var", "innovation_max"],
            "additionalProperties": False
        }
    },
    "required": ["mode", "kf"],
    "additionalProperties": False
}

def _clean_base_url(u: str) -> str:
    u = (u or "http://127.0.0.1:11434").strip()
    if "://" not in u:
        u = "http://" + u
    p = urlparse(u)
    return f"{p.scheme}://{p.netloc}".rstrip("/")

def _last_user_content(messages: List[Dict[str, str]]) -> str:
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if last_user:
        return last_user.get("content", "")
    return "\n\n".join(m.get("content", "") for m in messages)

def _sanitize_model(tag: str) -> str:
    tag = (tag or "").strip()
    if not tag:
        return "llama3.1:8b-instruct-q4_K_M"
    if tag.startswith("ollama ") or " " in tag:
        tag = tag.split()[-1]
    return tag

def _concat_messages(messages: List[Dict[str, str]]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "user")
        parts.append(f"[{role.upper()}]\n{m.get('content','').strip()}")
    return "\n\n".join(parts).strip()

def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t

def _repair_json(text: str) -> str:
    t = text.strip()

    import re
    t = re.sub(r',(\s*[}\]])', r'\1', t)
    t = re.sub(r'(["\d}])\s*\n\s*"', r'\1,\n  "', t)
    t = re.sub(r'(\w+)\s*,\s*\n\s*"', r'\1,\n  "', t)

    def fix_sci_notation(match):
        try:
            num = float(match.group(0))
            if abs(num) < 0.001 or abs(num) > 1000000:
                return str(num)
            return f"{num:.6f}".rstrip('0').rstrip('.')
        except:
            return match.group(0)

    t = re.sub(r'\d+\.\d+e[+-]\d+', fix_sci_notation, t)
    t = re.sub(r'"process_var"\s*:\s*,', '"process_var": 0.5,', t)
    t = re.sub(r'"meas_var"\s*:\s*,', '"meas_var": 0.1,', t)
    t = re.sub(r'"innovation_max"\s*:\s*,', '"innovation_max": 1.5,', t)

    return t

def ollama_caller(messages: List[Dict[str, str]], options: Dict[str, Any]) -> str:
    if not messages:
        raise ValueError("messages must contain at least one item")

    base_url = _clean_base_url(options.get("base_url", "http://127.0.0.1:11434"))
    model = _sanitize_model(options.get("model", "llama3.1:8b-instruct-q4_K_M"))

    user_read = options.get("timeout", None)
    read_timeout = float(user_read) if user_read is not None else 300.0
    read_timeout = max(read_timeout, 60.0)
    connect_timeout = 5.0
    timeout = (connect_timeout, read_timeout)

    max_attempts = int(options.get("max_attempts", 3))
    backoff = float(options.get("backoff", 1.0))

    s = requests.Session()
    s.trust_env = False

    def _post(url: str, payload: dict) -> dict:
        r = s.post(url, json=payload, timeout=timeout)
        if r.status_code >= 400:
            snippet = r.text[:500] if r.text else f"HTTP {r.status_code}"
            raise RuntimeError(f"{url} -> HTTP {r.status_code}: {snippet}")
        return r.json()

    def _attempt(callable_, *args, **kwargs):
        nonlocal backoff, timeout
        last_exc = None
        for attempt in range(1, max_attempts + 1):
            try:
                return callable_(*args, **kwargs)
            except requests.exceptions.ReadTimeout:
                if attempt < max_attempts:
                    timeout = (timeout[0], min(timeout[1] * 2.0, 600.0))
                    continue
                raise
            except requests.HTTPError as e:
                last_exc = e
                break
            except requests.RequestException as e:
                last_exc = e
                if attempt < max_attempts:
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, 5.0)
                    continue
                raise
            except Exception as e:
                last_exc = e
                if attempt < max_attempts:
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, 5.0)
                    continue
                raise
        if last_exc:
            raise last_exc

    enhanced_messages = messages.copy()
    for i, msg in enumerate(enhanced_messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if "Window summary" in content:
                reminder = (
                    "\nIMPORTANT: Your output MUST use these parameter ranges STRICTLY:\n"
                    "- static:  process_var=[0.03,0.05], meas_var=[0.10,0.18], innovation_max=[0.70,0.90]\n"
                    "- moving:  process_var=[0.30,1.00], meas_var=[0.04,0.12], innovation_max=[1.40,1.90]\n"
                    "- erratic: process_var=[0.80,1.50], meas_var=[0.12,0.25], innovation_max=[1.80,2.00]\n"
                )
                enhanced_messages[i]["content"] = content + reminder
                break

    content = _last_user_content(enhanced_messages)
    prompt = _concat_messages(enhanced_messages)

    def _parse_chat_api(data: dict) -> str:
        return _strip_code_fences(((data.get("message") or {}).get("content") or "").strip())

    def _parse_generate_api(data: dict) -> str:
        return _strip_code_fences((data.get("response") or "").strip())

    def _parse_openai_chat(data: dict) -> str:
        choices = data.get("choices") or []
        if choices and isinstance(choices, list):
            msg = (choices[0].get("message") or {})
            return _strip_code_fences((msg.get("content") or "").strip())
        return ""

    def _parse_openai_completions(data: dict) -> str:
        choices = data.get("choices") or []
        if choices and isinstance(choices, list):
            return _strip_code_fences((choices[0].get("text") or "").strip())
        return ""

    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "format": SCHEMA,
        "stream": False,
        "keep_alive": "5m",
    }
    try:
        data = _attempt(_post, url, payload)
        text = _parse_chat_api(data)
        text = _repair_json(text)
        json.loads(text)
        return text
    except Exception as e1:
        try:
            if "format of type string" in str(e1):
                payload["format"] = "json"
                data = _attempt(_post, url, payload)
                text = _parse_chat_api(data)
                text = _repair_json(text)
                json.loads(text)
                return text
        except Exception:
            pass

    try:
        data = _attempt(_post, f"{base_url}/v1/chat/completions", {
            "model": model,
            "messages": messages,
            "temperature": float(options.get("temperature", 0.0)),
            "max_tokens": int(options.get("max_tokens", 150)),
            "stream": False,
        })
        text = _parse_openai_chat(data)
        text = _repair_json(text)
        json.loads(text)
        return text
    except Exception:
        pass

    try:
        data = _attempt(_post, f"{base_url}/api/generate", {
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "keep_alive": "5m",
        })
        text = _parse_generate_api(data)
        text = _repair_json(text)
        json.loads(text)
        return text
    except Exception:
        pass

    try:
        data = _attempt(_post, f"{base_url}/v1/completions", {
            "model": model,
            "prompt": prompt,
            "temperature": float(options.get("temperature", 0.0)),
            "max_tokens": int(options.get("max_tokens", 150)),
            "stream": False,
        })
        text = _parse_openai_completions(data)
        text = _repair_json(text)
        json.loads(text)
        return text
    except Exception:
        pass

    data = _attempt(_post, f"{base_url}/api/generate", {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "5m",
    })
    text = _parse_generate_api(data)
    text = _repair_json(text)
    json.loads(text)
    return text
