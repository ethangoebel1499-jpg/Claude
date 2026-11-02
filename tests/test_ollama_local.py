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
