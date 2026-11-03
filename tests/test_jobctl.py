import io
import subprocess
import sys
import threading
import time

import pytest

from streamlearn import jobctl

# a fake job: its command line names the log like a real `python -m streamlearn.job <log> ...`; it starts a
# worker (which starts its own child) and a stand-in for the shared AI engine
JOB = """
import subprocess, sys, time
worker = "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); time.sleep(120)"
subprocess.Popen([sys.executable, "-c", worker])
subprocess.Popen([sys.argv[-1], "-c", "import time; time.sleep(120)", "ai-engine-stand-in"])
time.sleep(120)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="process test written for Mac/Linux")
def test_stop_ends_the_job_and_its_children_but_not_the_ai(tmp_path):
    log = tmp_path / "123.log"
    log.write_text("chunk 1/3\n")
    engine = tmp_path / "ollama"                 # a program named like the AI engine
    engine.symlink_to(sys.executable)
    p = subprocess.Popen([sys.executable, "-c", JOB, "-m", "streamlearn.job", str(log), "add", "x.txt", str(engine)])
    try:
        for _ in range(50):
            if len(jobctl.job_pids(log)) >= 3:
                break
            time.sleep(0.1)
        pids = jobctl.job_pids(log)
        assert len(pids) == 3                                 # job, worker, worker's child - not the AI engine
        ai = [pid for pid, _, cmd in jobctl.processes() if jobctl.is_ai_engine(cmd) and "ai-engine-stand-in" in cmd]
        assert ai and ai[0] not in pids
        assert jobctl.stop(log) == 3
        p.wait(timeout=10)
        time.sleep(0.5)
        assert not jobctl.running(log)
        assert jobctl.alive(ai[0])                            # the shared AI engine keeps running
        assert "Stopped by you" in log.read_text() and "[finished with exit code" in log.read_text()
    finally:
        p.kill()
        for pid, _, cmd in jobctl.processes():
            if jobctl.is_ai_engine(cmd) and "ai-engine-stand-in" in cmd:
                jobctl._kill(pid, hard=True)


def test_only_the_job_itself_is_matched(tmp_path):
    log = tmp_path / "9.log"
    table = [(10, 1, f"python -m streamlearn.job {log} add x.txt"), (11, 10, "python -u -m streamlearn add x.txt"),
             (12, 1, f"bash -c 'cat {log}; echo streamlearn.job'"), (13, 1, f"python -m streamlearn.job {log}2 add"),
             (14, 1, f'C:\\py\\python.exe -m apprentice.job "{log}" teach url')]
    assert sorted(jobctl.job_pids(log, table)) == [10, 11, 14]
    assert jobctl.is_ai_engine('"C:\\Users\\me\\AppData\\ollama.exe" serve') and jobctl.is_ai_engine("/x/ollama serve")
    assert not jobctl.is_ai_engine("python -m streamlearn add 'ollama tips.txt'")


def test_only_one_ai_job_at_a_time(tmp_path, monkeypatch):
    monkeypatch.setattr(jobctl, "LOCK", tmp_path / "ai.lock")
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    jobctl.LOCK.write_text(f"{holder.pid} 0")
    out, got = io.StringIO(), []
    t = threading.Thread(target=lambda: (jobctl.wait_turn(out, poll=0.1), got.append(True)), daemon=True)
    t.start()
    time.sleep(0.5)
    assert not got and "Waiting for the job before it" in out.getvalue()
    holder.kill()
    holder.wait()
    t.join(timeout=5)
    assert got                                            # the turn passes once the other job is gone
    jobctl.release()
    assert not jobctl.LOCK.exists()


def test_stale_lock_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(jobctl, "LOCK", tmp_path / "ai.lock")
    jobctl.LOCK.write_text("999999999 0")                 # a job that no longer exists
    jobctl.wait_turn(io.StringIO(), poll=0.1)
    assert jobctl.LOCK.read_text().split()[0] != "999999999"
    jobctl.release()


def test_delete_stream_removes_everything(tmp_path):
    from streamlearn.store import Store
    st = Store(tmp_path / "s.db")
    sid = st.add_stream("T", "2026-10-28", "inbox/x.txt", [(0, "hi"), (5, "there")], "now")
    cid = st.add_call(sid, {"t": 1, "said_at": "x", "asset": "BTC", "direction": "long", "kind": "plan",
                            "timeframe": "4h", "entry": 1.0, "stop": 0.9, "targets": [1.2], "conditions": [],
                            "quote": "q"})
    st.add_rule(sid, {"t": 1, "category": "entry", "rule": "r", "quote": "q"})
    st.review(cid, "correct")
    other = st.add_stream("Keep", "2026-10-29", "inbox/y.txt", [(0, "x")], "now")
    assert st.streams_from("inbox/x.txt") == [sid]
    st.delete_stream(sid)
    for t in ("streams", "segments", "calls", "rules", "reviews"):
        df = st.table(t)
        assert sid not in set(df.get("stream_id", df.get("id", []))) or t == "reviews"
    assert st.table("reviews").empty and list(st.table("streams").id) == [other]


def test_apprentice_delete_job_lessons(tmp_path, monkeypatch):
    from apprentice.store import Store
    st = Store(tmp_path / "a.db")
    monkeypatch.setenv("APPRENTICE_JOB", "job1")
    a = st.add_lesson("Video A", "u", 3, "vidA")
    st.add_setup({"name": "s", "direction": "long", "rules": [{"rule": "hammer"}], "lesson_id": a, "status": "active"})
    st.add_setup({"name": "p", "direction": "long", "rules": [{"rule": "hammer"}], "lesson_id": a, "status": "pinned"})
    monkeypatch.setenv("APPRENTICE_JOB", "job2")
    st.add_lesson("Video B", "u", 3, "vidB")
    st.job_stopped("job1")
    assert st.table("lessons").set_index("title").status["Video A"] == "stopped"
    assert not st.has_video("vidA")                       # pasting the playlist again redoes it
    assert st.forget_job("job1") == 1
    assert list(st.table("lessons").title) == ["Video B"]
    assert [s["name"] for s in st.setups()] == ["p"]          # pinned setups stay
