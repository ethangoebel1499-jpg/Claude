import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from streamlearn import ollama_local as o


def test_model_sized_for_memory():
    assert o.pick_model("auto", ram=8) == "qwen2.5:3b"
    assert o.pick_model(None, ram=16) == "qwen2.5:7b"
    assert o.pick_model("auto", ram=32) == "qwen2.5:14b"
    assert o.pick_model("qwen2.5:14b", ram=16) == "qwen2.5:7b"     # the old default is downsized to fit
    assert o.pick_model("llama3.1:8b", ram=8) == "llama3.1:8b"      # a model you chose is kept


@pytest.fixture
def ai_dir(tmp_path, monkeypatch):
    for name, path in (("DIR", tmp_path), ("MODELS", tmp_path / "models"), ("STATUS", tmp_path / "status.json"),
                       ("LOCK", tmp_path / "setup.lock")):
        monkeypatch.setattr(o, name, path)
    return tmp_path


def fake_ollama():
    models = []

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = {"/api/version": {"version": "test"}, "/api/tags": {"models": [{"name": m} for m in models]}}[self.path]
            self.send_response(200); self.end_headers(); self.wfile.write(json.dumps(body).encode())

        def do_POST(self):
            name = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["model"]
            self.send_response(200); self.end_headers()
            for done in (0, 50, 100):
                self.wfile.write((json.dumps({"total": 100, "completed": done}) + "\n").encode())
            models.append(name)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, models


def test_existing_server_gets_the_model(ai_dir, monkeypatch):
    srv, models = fake_ollama()
    monkeypatch.setattr(o, "ram_gb", lambda: 16)
    url, model, proc = o.ensure("auto", f"http://127.0.0.1:{srv.server_port}", log=lambda *a: None)
    assert model == "qwen2.5:7b" and models == ["qwen2.5:7b"] and proc is None
    assert o.status()["ready"] and not o.LOCK.exists()
    o.ensure("auto", url, log=lambda *a: None)
    assert models == ["qwen2.5:7b"]                                  # not downloaded twice
    srv.shutdown()


def test_failure_is_reported_not_hidden(ai_dir, monkeypatch):
    monkeypatch.setattr(o, "start_server", lambda url, log: (_ for _ in ()).throw(RuntimeError("no network")))
    with pytest.raises(RuntimeError):
        o.ensure("auto", "http://127.0.0.1:9", log=lambda *a: None)
    assert "no network" in o.status()["error"] and not o.LOCK.exists()


def test_falls_back_to_hugging_face_when_ollama_download_keeps_failing(ai_dir, monkeypatch):
    blobs, created, served = {}, [], {"drops": 1}
    payload = b"GGUF" + bytes(range(256)) * 400

    class Ollama(BaseHTTPRequestHandler):
        def do_GET(self):
            body = {"/api/version": {"version": "t"}, "/api/tags": {"models": [{"name": m} for m in created]}}[self.path]
            self.send_response(200); self.end_headers(); self.wfile.write(json.dumps(body).encode())

        def do_HEAD(self):
            self.send_response(200 if self.path.split("/")[-1] in blobs else 404); self.end_headers()

        def do_POST(self):
            data = self.rfile.read(int(self.headers["Content-Length"]))
            if self.path == "/api/pull":
                self.send_response(200); self.end_headers()
                self.wfile.write(b'{"error": "pull model manifest: connection forcibly closed"}\n')
            elif self.path.startswith("/api/blobs/"):
                blobs[self.path.split("/")[-1]] = data
                self.send_response(201); self.end_headers()
            else:                                           # /api/create
                req = json.loads(data)
                assert list(req["files"].values())[0] in blobs
                created.append(req["model"])
                self.send_response(200); self.end_headers(); self.wfile.write(b'{"status": "success"}')

        def log_message(self, *a):
            pass

    class HuggingFace(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/api/models/"):
                body = json.dumps({"siblings": [{"rfilename": "README.md"},
                                                {"rfilename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf"}]}).encode()
                self.send_response(200); self.end_headers(); self.wfile.write(body)
                return
            start = int(self.headers.get("Range", "bytes=0-")[6:-1] or 0)
            chunk = payload[start:]
            if served["drops"]:                                # cut the first download off halfway
                served["drops"] -= 1
                chunk = chunk[: len(chunk) // 2]
                self.send_response(200); self.send_header("Content-Length", str(len(payload))); self.end_headers()
                self.wfile.write(chunk)
                return
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(chunk))); self.end_headers(); self.wfile.write(chunk)

        def log_message(self, *a):
            pass

    servers = [HTTPServer(("127.0.0.1", 0), h) for h in (Ollama, HuggingFace)]
    for srv in servers:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(o, "HF", f"http://127.0.0.1:{servers[1].server_port}")
    monkeypatch.setattr(o.time, "sleep", lambda s: None)
    url = f"http://127.0.0.1:{servers[0].server_port}"
    o.get_model(url, "qwen2.5:7b", log=lambda *a: None)
    assert created == ["qwen2.5:7b"]
    assert list(blobs.values())[0] == payload                  # resumed download is byte-for-byte complete
    assert not list((ai_dir / "gguf").glob("*.gguf"))           # the downloaded copy is cleaned up
    for srv in servers:
        srv.shutdown()
