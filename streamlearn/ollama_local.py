"""Set up the local AI model for the stream agent by itself - no installing, no commands.

ensure() makes sure an Ollama server is answering and has a model:
  1. an Ollama already running on this computer is used as is
  2. otherwise Ollama is downloaded once into runtime/ollama/ (nothing system-wide) and started in
     the background, keeping its models in runtime/ollama/models/
  3. the model is chosen for this computer's memory (unless you picked one in Account) and
     downloaded once
Progress goes to runtime/ollama/status.json, which the app window and the dashboard show.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "runtime" / "ollama"
MODELS = DIR / "models"
STATUS = DIR / "status.json"
LOCK = DIR / "setup.lock"
RELEASE = "https://github.com/ollama/ollama/releases/latest/download/"
HF = os.environ.get("PAPERTRADER_HF", "https://huggingface.co")
# the same models as plain files on Hugging Face, for when Ollama's own model server won't download
HF_REPOS = {"qwen2.5:0.5b": "bartowski/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5:3b": "bartowski/Qwen2.5-3B-Instruct-GGUF",
            "qwen2.5:7b": "bartowski/Qwen2.5-7B-Instruct-GGUF", "qwen2.5:14b": "bartowski/Qwen2.5-14B-Instruct-GGUF"}
DEFAULT_URL = "http://127.0.0.1:11434"
OLD_DEFAULT = "qwen2.5:14b"          # what Account used to save by default


# ---- choosing a model --------------------------------------------------------------------
def ram_gb() -> float:
    try:
        if os.name == "nt":
            import ctypes

            class MEM(ctypes.Structure):
                _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                            ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ext", ctypes.c_ulonglong)]
            m = MEM()
            m.len = ctypes.sizeof(MEM)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.total / 2**30
        if sys.platform == "darwin":
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 2**30
        with open("/proc/meminfo") as f:
            return int(f.readline().split()[1]) / 2**20
    except Exception:
        return 8.0


def auto_model(ram: float | None = None) -> str:
    ram = ram_gb() if ram is None else ram
    return "qwen2.5:14b" if ram >= 20 else "qwen2.5:7b" if ram >= 10 else "qwen2.5:3b"


def pick_model(setting: str | None, ram: float | None = None) -> str:
    """'auto' (or nothing) -> sized for this computer; the old 14b default is downsized if it won't fit."""
    ram = ram_gb() if ram is None else ram
    if not setting or setting == "auto" or (setting == OLD_DEFAULT and ram < 20):
        return auto_model(ram)
    return setting


# ---- status the window / dashboard show ------------------------------------------------------
def set_status(text: str, pct: float | None = None, ready: bool = False, error: str | None = None, **extra) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({"text": text, "pct": pct, "ready": ready, "error": error, "time": time.time(),
                                  **extra}))


def status() -> dict:
    try:
        return json.loads(STATUS.read_text())
    except Exception:
        return {}


# ---- the server ------------------------------------------------------------------------------
def _get(url: str, timeout: float = 2) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except Exception:
        return None


def server_up(url: str) -> bool:
    return _get(f"{url}/api/version") is not None


def exe() -> Path | None:
    name = "ollama.exe" if os.name == "nt" else "ollama"
    found = sorted(DIR.rglob(name)) if DIR.exists() else []
    found = [f for f in found if f.is_file() and "models" not in f.parts]
    return found[0] if found else None


def _asset() -> str:
    arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "amd64"
    if os.name == "nt":
        return f"ollama-windows-{arch}.zip"
    if sys.platform == "darwin":
        return "ollama-darwin.tgz"
    return f"ollama-linux-{arch}.tar.zst"


def download(log=print) -> Path:
    """Fetch Ollama for this computer into runtime/ollama (once)."""
    DIR.mkdir(parents=True, exist_ok=True)
    name = _asset()
    part = DIR / name
    log(f"Downloading the local AI engine (Ollama, {name})...")
    _resumable(RELEASE + name, part, "Downloading the AI engine")
    set_status("Unpacking the AI engine...")
    target = DIR / "engine"
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir()
    if name.endswith(".zip"):
        with zipfile.ZipFile(part) as z:
            z.extractall(target)
    elif name.endswith(".tgz"):
        with tarfile.open(part) as t:
            t.extractall(target, filter="data")
    else:                                               # .tar.zst (Linux)
        try:
            import zstandard
            with open(part, "rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as z, \
                    tarfile.open(fileobj=z, mode="r|") as t:
                t.extractall(target, filter="data")
        except ImportError:
            subprocess.run(["tar", "--zstd", "-xf", str(part), "-C", str(target)], check=True)
    part.unlink(missing_ok=True)
    found = exe()
    if not found:
        raise RuntimeError("the Ollama download didn't contain the program")
    if os.name != "nt":
        found.chmod(0o755)
    return found


def start_server(url: str, log=print) -> subprocess.Popen:
    program = exe() or download(log)
    host = url.split("//", 1)[-1]
    env = {**os.environ, "OLLAMA_HOST": host, "OLLAMA_MODELS": str(MODELS)}
    MODELS.mkdir(parents=True, exist_ok=True)
    out = open(DIR / "server.log", "a", encoding="utf-8", errors="replace")
    proc = subprocess.Popen([str(program), "serve"], env=env, stdout=out, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, creationflags=0x08000000 if os.name == "nt" else 0)
    for _ in range(120):
        if server_up(url):
            log("Local AI engine running")
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"the AI engine stopped right away (see {DIR / 'server.log'})")
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("the AI engine didn't start within a minute")


def has_model(url: str, model: str) -> bool:
    tags = _get(f"{url}/api/tags", timeout=5) or {}
    names = {m.get("name", "") for m in tags.get("models", [])}
    return model in names or (":" not in model and f"{model}:latest" in names)


def pull(url: str, model: str, log=print) -> None:
    log(f"Downloading the AI model {model} (only once)...")
    req = urllib.request.Request(f"{url}/api/pull", data=json.dumps({"model": model, "stream": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    last = 0.0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            msg = json.loads(line or b"{}")
            if msg.get("error"):
                raise RuntimeError(f"model download failed: {msg['error']}")
            total, done = msg.get("total") or 0, msg.get("completed") or 0
            if total and time.time() - last > 1:
                last = time.time()
                set_status(f"Downloading the AI model {model}: {done / 2**30:.1f} of {total / 2**30:.1f} GB",
                           pct=done / total * 100)
    if not has_model(url, model):
        raise RuntimeError(f"model {model} didn't finish downloading")


# ---- when Ollama's model server won't download (it was cut off on some home networks) -----------
class ipv4_only:
    """Connect over IPv4 only (a network that drops IPv6 connections to a host then still works)."""
    def __enter__(self):
        import socket
        self.socket, self.orig = socket, socket.getaddrinfo
        def v4(host, *a, **k):
            res = self.orig(host, *a, **k)
            return [r for r in res if r[0] == socket.AF_INET] or res
        socket.getaddrinfo = v4

    def __exit__(self, *exc):
        self.socket.getaddrinfo = self.orig


def _hf_file(repo: str) -> str:
    """The single-file 4-bit (Q4_K_M) model in a Hugging Face repo."""
    with urllib.request.urlopen(f"{HF}/api/models/{repo}", timeout=30) as r:
        files = [x["rfilename"] for x in json.loads(r.read()).get("siblings", [])]
    pick = [f for f in files if f.lower().endswith("q4_k_m.gguf") and "-of-" not in f]
    if not pick:
        raise RuntimeError(f"no Q4_K_M model file in {repo}")
    return pick[0]


def _resumable(url: str, dest: Path, label: str, attempts: int = 6) -> None:
    """Download with resume: a dropped connection continues where it stopped."""
    part = dest.with_name(dest.name + ".part")
    for attempt in range(attempts):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "paper-trader", **({"Range": f"bytes={have}-"} if have else {})})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:            # server ignored the resume: start over
                    have = 0
                total = have + int(r.headers.get("Content-Length") or 0)
                with open(part, "ab" if have else "wb") as out:
                    done, last = have, 0.0
                    while chunk := r.read(1 << 20):
                        out.write(chunk)
                        done += len(chunk)
                        if total and time.time() - last > 1:
                            last = time.time()
                            set_status(f"{label}: {done / 2**30:.2f} of {total / 2**30:.2f} GB", pct=done / total * 100)
            if total and done < total:              # connection dropped early: resume, don't keep a cut-off file
                raise ConnectionError(f"download stopped at {done:,} of {total:,} bytes")
            part.replace(dest)
            return
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(10 * (attempt + 1))


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def from_hugging_face(url: str, model: str, log=print) -> None:
    """Download the model file from Hugging Face (IPv4, resumable) and add it to Ollama as `model`."""
    repo = HF_REPOS[model]
    with ipv4_only():
        name = _hf_file(repo)
        dest = DIR / "gguf" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            log(f"Downloading {model} from Hugging Face instead ({name})...")
            _resumable(f"{HF}/{repo}/resolve/main/{name}", dest, f"Downloading the AI model {model} (Hugging Face)")
    set_status(f"Adding {model} to the local AI...")
    digest = "sha256:" + _sha256(dest)
    try:
        urllib.request.urlopen(urllib.request.Request(f"{url}/api/blobs/{digest}", method="HEAD"), timeout=30)
    except urllib.error.HTTPError:
        with open(dest, "rb") as f:
            req = urllib.request.Request(f"{url}/api/blobs/{digest}", data=f, method="POST",
                                         headers={"Content-Length": str(dest.stat().st_size)})
            urllib.request.urlopen(req, timeout=3600).read()
    body = json.dumps({"model": model, "files": {name: digest}, "stream": False}).encode()
    with urllib.request.urlopen(urllib.request.Request(f"{url}/api/create", data=body,
                                                       headers={"Content-Type": "application/json"}), timeout=3600) as r:
        res = json.loads(r.read() or b"{}")
    if res.get("error") or not has_model(url, model):
        raise RuntimeError(f"couldn't add the downloaded model: {res.get('error', 'not listed afterwards')}")
    dest.unlink(missing_ok=True)                    # Ollama keeps its own copy


def get_model(url: str, model: str, log=print, tries: int = 3, wait: float = 15) -> None:
    """Ollama's own download, retried; then the same model from Hugging Face."""
    err = None
    for attempt in range(tries):
        try:
            return pull(url, model, log)
        except Exception as e:
            err = e
            if attempt < tries - 1:
                log(f"Model download interrupted ({str(e)[:120]}) - retrying...")
                set_status(f"Model download interrupted - retrying ({attempt + 2} of {tries})...")
                time.sleep(wait * (attempt + 1))
    if model not in HF_REPOS:
        raise err
    log("Ollama's model server keeps cutting the download off on this network - using Hugging Face instead")
    from_hugging_face(url, model, log)


# ---- the one call everything uses -------------------------------------------------------------
def ensure(setting_model: str | None = None, url: str | None = None, log=print) -> tuple[str, str, subprocess.Popen | None]:
    """Make the local AI ready. Returns (server url, model, server process if this call started it)."""
    url = (url or DEFAULT_URL).rstrip("/").replace("localhost", "127.0.0.1")
    model = pick_model(setting_model)
    DIR.mkdir(parents=True, exist_ok=True)
    # one setup at a time (the app and a stream job can both ask): wait for the other one instead
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        if time.time() - LOCK.stat().st_mtime < 3 * 3600:
            log("The local AI is being set up by the app - waiting for it...")
            while LOCK.exists() and time.time() - status().get("time", 0) < 600:   # gave up if silent 10 min
                time.sleep(3)
            s = status()
            if s.get("ready") and server_up(s.get("url", url)):
                return s.get("url", url), s.get("model", model), None
            if s.get("error"):
                raise RuntimeError(s["error"])
        LOCK.unlink(missing_ok=True)
        return ensure(setting_model, url, log)
    proc = None
    try:
        if not server_up(url):
            set_status("Starting the local AI engine...")
            proc = start_server(url, log)
        if not has_model(url, model):
            get_model(url, model, log)
        set_status(f"Ready ({model})", pct=100, ready=True, url=url, model=model)
        return url, model, proc
    except Exception as e:
        set_status("Local AI setup failed", error=f"{type(e).__name__}: {e}")
        if proc:
            proc.terminate()
        raise
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    u, m, p = ensure()
    print(f"ready: {m} at {u}")
