"""Paper Trader - the desktop app.

Double-click the "Paper Trader" shortcut (setup makes one on the Desktop and in the Start menu).
It checks GitHub for updates and installs them, starts the paper trader and the dashboard in the
background (no console windows), opens the dashboard in your browser, and keeps checking for
updates every few hours. Closing the window stops paper trading; open positions are saved and
pick up next time.
"""

from __future__ import annotations

import hashlib
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
import tomllib
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import updater  # noqa: E402

PORT = 8501
LOCK_PORT = 47651                        # one Paper Trader at a time
UPDATE_EVERY = 6 * 3600
NO_WINDOW = 0x08000000 if os.name == "nt" else 0
SELF_FILES = ("launcher.pyw", "updater.py", "bootstrap.pyw")


def python_exe() -> str:
    """The app's Python (the .venv one), with a console-less sibling preferred for children."""
    if os.environ.get("PAPERTRADER_PY"):
        return os.environ["PAPERTRADER_PY"]
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and (exe.parent / "python.exe").exists():
        return str(exe.parent / "python.exe")
    return str(exe)


def lan_address() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.168.1.1", 9))
            return s.getsockname()[0]
    except OSError:
        return "this computer's IP"


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def files_hash(names) -> str:
    h = hashlib.sha256()
    for n in names:
        p = ROOT / n
        h.update(p.read_bytes() if p.exists() else b"")
    return h.hexdigest()


class Services:
    """The trader and the dashboard, as background processes."""

    def __init__(self, log):
        self.log, self.py = log, python_exe()
        self.trader = self.dash = self.ai = None
        self.phone = False
        self.want_trading = True
        self._last_restart = 0.0

    def _config(self) -> tuple[str, str]:
        cfg = ROOT / "live.toml"
        if not cfg.exists():
            shutil.copy(ROOT / "livepaper" / "live.example.toml", cfg)
            self.log("Created live.toml (your paper-trading runs)")
        try:
            with open(cfg, "rb") as f:
                db = tomllib.load(f).get("app", {}).get("db", "livepaper.db")
        except (OSError, tomllib.TOMLDecodeError) as e:
            self.log(f"live.toml has a mistake ({e}) - fix it or delete it to get a fresh one")
            db = "livepaper.db"
        return str(cfg), db

    def _spawn(self, args, logname, env=None):
        out = open(ROOT / logname, "a", buffering=1, encoding="utf-8", errors="replace")
        return subprocess.Popen([self.py, *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=env, creationflags=NO_WINDOW)

    def start_trader(self):
        if self.trader and self.trader.poll() is None:
            return
        self.want_trading = True
        self._config()
        self.trader = self._spawn(["-m", "livepaper", "run"], "trader.log")
        self.log("Paper trading started")

    def stop_trader(self):
        self.want_trading = False
        self._stop(self.trader)
        self.trader = None
        self.log("Paper trading stopped")

    def start_dashboard(self):
        if self.dash and self.dash.poll() is None:
            return
        cfg, db = self._config()
        env = {**os.environ, "LIVEPAPER_DB": db, "LIVEPAPER_CONFIG": cfg}
        args = ["-m", "streamlit", "run", str(ROOT / "livepaper" / "dashboard.py"), "--server.port", str(PORT),
                "--server.headless", "true", "--browser.gatherUsageStats", "false"]
        if self.phone:
            args += ["--server.address", "0.0.0.0"]
        self.dash = self._spawn(args, "dashboard.log", env)

    def stop_dashboard(self):
        self._stop(self.dash)
        self.dash = None

    @staticmethod
    def _stop(p):
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()

    def stop_all(self):
        self._stop(self.trader)
        self._stop(self.dash)
        self.trader = self.dash = None

    def stop_ai(self):
        self._stop(self.ai)
        self.ai = None

    def trading(self) -> bool:
        return bool(self.trader and self.trader.poll() is None)

    def dashboard_up(self) -> bool:
        return bool(self.dash and self.dash.poll() is None and port_open(PORT))

    def watchdog(self):
        """Restart the trader if it stopped on its own (at most once a minute)."""
        if self.want_trading and self.trader and self.trader.poll() is not None \
                and time.time() - self._last_restart > 60:
            self._last_restart = time.time()
            self.log("Trader stopped unexpectedly (see trader.log) - restarting it")
            self.trader = None
            self.start_trader()


class App:
    def __init__(self, lock: socket.socket):
        import tkinter as tk
        from tkinter import scrolledtext, ttk
        self.tk, self.lock = tk, lock
        self.q: queue.Queue = queue.Queue()
        self.svc = Services(self.log)
        self.busy = False
        self.exit_code = 0

        if os.name == "nt":                   # taskbar shows the app's own icon, not Python's
            try:
                import ctypes
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("PaperTrader.App")
            except Exception:
                pass
        self.root = root = tk.Tk()
        root.title("Paper Trader")
        root.geometry("760x540")
        root.minsize(520, 420)
        root.configure(bg="#0f172a")
        try:
            if os.name == "nt":
                root.iconbitmap(str(ROOT / "assets" / "icon.ico"))
            self._icon = tk.PhotoImage(file=str(ROOT / "assets" / "icon_256.png"))
            root.iconphoto(True, self._icon)
            self._logo = self._icon.subsample(4, 4)
        except Exception:
            self._logo = None

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TButton", padding=(12, 7), font=("Segoe UI", 10))
        style.configure("Accent.TButton", background="#16a34a", foreground="white")
        style.map("Accent.TButton", background=[("active", "#15803d")])
        style.configure("TCheckbutton", background="#0f172a", foreground="#cbd5e1", font=("Segoe UI", 10))
        style.map("TCheckbutton", background=[("active", "#0f172a")])

        head = tk.Frame(root, bg="#0f172a")
        head.pack(fill="x", padx=18, pady=(16, 6))
        if self._logo:
            tk.Label(head, image=self._logo, bg="#0f172a").pack(side="left", padx=(0, 12))
        titles = tk.Frame(head, bg="#0f172a")
        titles.pack(side="left")
        tk.Label(titles, text="Paper Trader", font=("Segoe UI", 20, "bold"), fg="white", bg="#0f172a").pack(anchor="w")
        self.version = tk.Label(titles, text="", font=("Segoe UI", 9), fg="#94a3b8", bg="#0f172a")
        self.version.pack(anchor="w")

        card = tk.Frame(root, bg="#1e293b")
        card.pack(fill="x", padx=18, pady=8)
        self.rows = {}
        for i, (key, label) in enumerate([("trading", "Paper trading"), ("dash", "Dashboard"), ("upd", "Updates"),
                                          ("ai", "Stream AI")]):
            tk.Label(card, text=label, font=("Segoe UI", 10), fg="#94a3b8", bg="#1e293b").grid(
                row=i, column=0, sticky="w", padx=14, pady=6)
            v = tk.Label(card, text="…", font=("Segoe UI", 10, "bold"), fg="white", bg="#1e293b", cursor="")
            v.grid(row=i, column=1, sticky="w", padx=6, pady=6)
            self.rows[key] = v
        self.rows["dash"].bind("<Button-1>", lambda e: self.open_dashboard())

        btns = tk.Frame(root, bg="#0f172a")
        btns.pack(fill="x", padx=18, pady=6)
        ttk.Button(btns, text="Open dashboard", style="Accent.TButton", command=self.open_dashboard).pack(side="left")
        self.trade_btn = ttk.Button(btns, text="Stop trading", command=self.toggle_trading)
        self.trade_btn.pack(side="left", padx=8)
        ttk.Button(btns, text="Check for updates", command=lambda: self.run_update(manual=True)).pack(side="left")
        ttk.Button(btns, text="Open folder", command=self.open_folder).pack(side="left", padx=8)
        ttk.Button(btns, text="Undo last update", command=self.undo_update).pack(side="left")

        self.phone = tk.BooleanVar(value=False)
        ttk.Checkbutton(root, text="Phone access (same Wi-Fi)", variable=self.phone,
                        command=self.toggle_phone).pack(anchor="w", padx=18, pady=(4, 0))

        self.logbox = scrolledtext.ScrolledText(root, height=10, bg="#020617", fg="#cbd5e1", insertbackground="white",
                                                font=("Consolas", 9), relief="flat", state="disabled", wrap="word")
        self.logbox.pack(fill="both", expand=True, padx=18, pady=(10, 16))

        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(200, self.pump)
        root.after(2000, self.refresh)
        self.show_version()
        threading.Thread(target=self.startup, daemon=True).start()
        root.after(UPDATE_EVERY * 1000, self.periodic_update)

    # ---- plumbing -------------------------------------------------------------------
    def log(self, msg: str):
        self.q.put(f"{time.strftime('%H:%M:%S')}  {msg}")

    def pump(self):
        while not self.q.empty():
            line = self.q.get()
            if line == "__relaunch__":
                return self.relaunch()
            if callable(line):                    # window changes asked for by background threads
                line()
                continue
            self.logbox.configure(state="normal")
            self.logbox.insert("end", line + "\n")
            self.logbox.see("end")
            self.logbox.configure(state="disabled")
        self.root.after(200, self.pump)

    def show_version(self):
        sha = updater.current()
        self.version.configure(text=f"version {sha[:7]}" if sha else "version: first update pending")

    def refresh(self):
        self.svc.watchdog()
        on = self.svc.trading()
        self.rows["trading"].configure(text="● Running" if on else "● Stopped", fg="#22c55e" if on else "#f87171")
        self.trade_btn.configure(text="Stop trading" if on else "Start trading")
        if self.svc.dashboard_up():
            addr = f"http://{lan_address()}:{PORT}" if self.svc.phone else f"http://localhost:{PORT}"
            self.rows["dash"].configure(text=f"● {addr}  (click to open)", fg="#38bdf8", cursor="hand2")
        else:
            self.rows["dash"].configure(text="● Starting…" if self.svc.dash else "● Off", fg="#fbbf24", cursor="")
        self.rows["ai"].configure(**ai_row())
        self.root.after(3000, self.refresh)

    # ---- startup / updates -------------------------------------------------------------
    def startup(self):
        if updater.pending():                 # restarted to run a new version of the app itself
            self.log("Checking that the new version works...")
        elif not self._update():
            return
        self.start_and_verify(first=True)

    def start_and_verify(self, first=False):
        """Start the trader and dashboard; an update still being checked is kept only if they come up."""
        try:
            self.svc.start_trader()
            self.svc.start_dashboard()
            ok, why = self._wait_healthy()
        except Exception as e:
            ok, why = False, f"{type(e).__name__}: {e}"
        if updater.pending():
            if ok:
                updater.confirm()
                sha = updater.current() or ""
                self.log(f"Update {sha[:7]} works - keeping it")
                self.set_upd(f"Updated to {sha[:7]}", "#22c55e")
            else:
                self.log(f"The new version has a problem: {why}")
                if not self._rollback(why, stop=True):
                    return
                self.svc.start_trader()
                self.svc.start_dashboard()
                ok, why = self._wait_healthy()
        if not ok:
            self.log(f"Problem starting: {why}")
        elif first:
            webbrowser.open(f"http://localhost:{PORT}")
            self.log("Dashboard open in your browser")
        if first:
            self.start_local_ai()

    def start_local_ai(self):
        """Download / start the stream agent's local AI in the background (only with the local model)."""
        cfg = app_settings()
        if cfg.get("stream_backend", "ollama") != "ollama":
            return
        def work():
            try:
                from streamlearn import ollama_local
                url, model, proc = ollama_local.ensure(cfg.get("ollama_model"), cfg.get("ollama_url"), self.log)
                self.svc.ai = proc
                self.log(f"Stream AI ready ({model})")
            except Exception as e:
                self.log(f"Stream AI setup failed: {e} - it retries next time")
        threading.Thread(target=work, daemon=True).start()

    def _wait_healthy(self) -> tuple[bool, str]:
        for _ in range(180):                  # dashboard answering within 90 s
            if self.svc.dashboard_up():
                break
            if self.svc.dash and self.svc.dash.poll() is not None:
                return False, "the dashboard stopped (see dashboard.log)"
            time.sleep(0.5)
        else:
            return False, "the dashboard didn't start within 90 seconds (see dashboard.log)"
        for _ in range(15):                   # trader still running 15 s later
            if self.svc.want_trading and not self.svc.trading():
                return False, "the trader stopped (see trader.log)"
            time.sleep(1)
        return True, ""

    def _update(self) -> bool:
        """Update from GitHub and check it. False if the app is restarting itself."""
        self.busy = True
        self.set_upd("Checking…", "#fbbf24")
        try:
            before = files_hash(SELF_FILES)
            res = updater.update(self.log)
        except Exception as e:
            self.log(f"Couldn't check for updates ({type(e).__name__}) - running the installed version")
            self.set_upd("Couldn't check (offline?)", "#f87171")
            self.busy = False
            return True
        self.log(res["message"])
        try:
            if not res["updated"]:
                self.set_upd(res["message"], "#22c55e" if res["message"] == "Up to date" else "#fbbf24")
                return True
            if res["deps_changed"] and not updater.install_deps(python_exe(), self.log):
                return self._rollback("its packages didn't install")
            self.log("Running the self-check on the new version...")
            ok, out = updater.healthcheck(python_exe())
            if not ok:
                self.log(out)
                return self._rollback("it failed the self-check")
            self.log("Self-check passed")
            if files_hash(SELF_FILES) != before:
                self.log("The app itself was updated - restarting it")
                self.q.put("__relaunch__")
                return False
            return True
        finally:
            self.busy = False

    def _rollback(self, reason: str, stop: bool = False) -> bool:
        """Go back to the version before the update. False if the app is restarting itself."""
        if stop:
            self.svc.stop_all()
        before = files_hash(SELF_FILES)
        res = updater.rollback(self.log, reason)
        if res.get("deps_changed"):
            updater.install_deps(python_exe(), self.log)
        self.set_upd(f"Last update didn't work - back on {(res.get('to') or '?')[:7]}", "#fbbf24")
        if files_hash(SELF_FILES) != before:
            self.q.put("__relaunch__")
            return False
        return True

    def set_upd(self, text, color):
        self.q.put(lambda: (self.rows["upd"].configure(text=f"● {text}", fg=color), self.show_version()))

    def run_update(self, manual=False):
        if self.busy:
            return
        def work():
            old = updater.current()
            if not self._update():
                return
            if updater.pending() and updater.current() != old:   # new code passed the self-check: run it
                self.log("Restarting the trader and dashboard on the new version")
                self.svc.stop_all()
                self.start_and_verify()
        threading.Thread(target=work, daemon=True).start()

    def periodic_update(self):
        self.run_update()
        self.root.after(UPDATE_EVERY * 1000, self.periodic_update)

    def undo_update(self):
        from tkinter import messagebox
        s = updater.state()
        if not s:
            messagebox.showinfo("Paper Trader", "There's no update to undo yet.")
            return
        if not messagebox.askyesno("Paper Trader", f"Go back to the version before the last update "
                                                   f"({(s['from'] or 'previous')[:7]})?"):
            return
        def work():
            if self._rollback("undone by you", stop=True):
                self.start_and_verify()
        threading.Thread(target=work, daemon=True).start()

    def relaunch(self):
        self.svc.stop_all()
        self.svc.stop_ai()
        self.lock.close()
        if os.environ.get("PAPERTRADER_BOOTSTRAP"):
            self.exit_code = 3                # bootstrap.pyw opens the app again
        else:
            subprocess.Popen([sys.executable, str(ROOT / "bootstrap.pyw")], cwd=ROOT, creationflags=NO_WINDOW)
        self.root.destroy()

    # ---- buttons ---------------------------------------------------------------------
    def open_dashboard(self):
        if not self.svc.dash:
            self.svc.start_dashboard()
        webbrowser.open(f"http://localhost:{PORT}")

    def toggle_trading(self):
        (self.svc.stop_trader if self.svc.trading() else self.svc.start_trader)()
        self.refresh()

    def toggle_phone(self):
        self.svc.phone = self.phone.get()
        self.svc.stop_dashboard()
        self.svc.start_dashboard()
        if self.svc.phone:
            self.log(f"Phone access on: open http://{lan_address()}:{PORT} on your phone (same Wi-Fi), "
                     "then Share -> Add to Home Screen")

    def open_folder(self):
        if os.name == "nt":
            os.startfile(ROOT)
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(ROOT)])

    def close(self):
        from tkinter import messagebox
        if self.svc.trading() and not messagebox.askyesno(
                "Paper Trader", "Closing Paper Trader stops paper trading (open positions are saved and resume "
                                "next time).\n\nClose it?"):
            return
        self.svc.stop_all()
        self.svc.stop_ai()
        self.root.destroy()


def app_settings() -> dict:
    f = Path(os.environ.get("LOCAL_SETTINGS", ROOT / "local_settings.json"))
    try:
        import json
        return json.loads(f.read_text()) if f.exists() else {}
    except Exception:
        return {}


def ai_row() -> dict:
    if app_settings().get("stream_backend", "ollama") != "ollama":
        return {"text": "● Claude (API)", "fg": "#22c55e"}
    from streamlearn import ollama_local
    s = ollama_local.status()
    if s.get("ready"):
        return {"text": f"● Ready ({s.get('model')})", "fg": "#22c55e"}
    if s.get("error"):
        return {"text": "● Setup failed - retries next start", "fg": "#f87171"}
    text = s.get("text") or "Not set up yet"
    return {"text": f"● {text}" + (f" ({s['pct']:.0f}%)" if s.get("pct") is not None and "%" not in text else ""),
            "fg": "#fbbf24"}


def main():
    lock = socket.socket()
    try:
        lock.bind(("127.0.0.1", LOCK_PORT))
    except OSError:                       # already running: just show the dashboard
        webbrowser.open(f"http://localhost:{PORT}")
        return
    app = App(lock)
    app.root.mainloop()
    sys.exit(app.exit_code)


if __name__ == "__main__":
    main()
