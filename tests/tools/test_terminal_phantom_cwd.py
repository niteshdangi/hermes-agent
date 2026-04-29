"""Regression test for phantom-CWD retry loop.

When the persistent shell environment's cached cwd is deleted out-of-band
(e.g. ``rm -rf /tmp/tirith-test``), every subsequent ``Popen(cwd=...)`` call
used to raise ``FileNotFoundError`` 3 times through the retry loop before
failing the command. The fix detects ENOENT on env.cwd at command-start, logs
once, and resets cwd to $HOME before dispatch.
"""

import json
import logging
import os
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from tools import terminal_tool as tt


def test_reset_env_cwd_if_missing_resets_to_home(caplog):
    missing = "/tmp/definitely-not-a-real-dir-phantom-cwd-test-xyz"
    if os.path.exists(missing):
        shutil.rmtree(missing, ignore_errors=True)

    env = SimpleNamespace(cwd=missing)
    tt._phantom_cwd_logged.clear()
    with caplog.at_level(logging.WARNING, logger=tt.logger.name):
        tt._reset_env_cwd_if_missing(env)

    assert env.cwd == os.path.expanduser("~")
    assert any("no longer exists" in rec.message for rec in caplog.records)


def test_reset_env_cwd_if_missing_no_op_when_present():
    with tempfile.TemporaryDirectory() as tmp:
        env = SimpleNamespace(cwd=tmp)
        tt._reset_env_cwd_if_missing(env)
        assert env.cwd == tmp


def test_reset_env_cwd_logs_only_once_per_path(caplog):
    env = SimpleNamespace(cwd="/tmp/another-phantom-cwd-test-abc")
    if os.path.exists(env.cwd):
        shutil.rmtree(env.cwd, ignore_errors=True)
    tt._phantom_cwd_logged.clear()

    with caplog.at_level(logging.WARNING, logger=tt.logger.name):
        tt._reset_env_cwd_if_missing(env)
        # Re-set to the missing path; second call should not re-log.
        env.cwd = "/tmp/another-phantom-cwd-test-abc"
        tt._reset_env_cwd_if_missing(env)

    warnings = [r for r in caplog.records if "no longer exists" in r.message]
    assert len(warnings) == 1


def test_terminal_tool_recovers_after_cwd_deleted(tmp_path):
    """End-to-end: deleting the persistent-shell cwd does not break next command."""
    doomed = tmp_path / "doomed"
    doomed.mkdir()

    # First: cd into it via terminal_tool so the persistent env caches it.
    task_id = "phantom-cwd-regression-test"
    try:
        result = json.loads(tt.terminal_tool(f"cd {doomed} && pwd", task_id=task_id))
        assert result["exit_code"] == 0
        assert str(doomed) in result["output"]

        # Now nuke it out from under the session.
        shutil.rmtree(doomed)

        # Next command must NOT FileNotFoundError; it should rebase to $HOME.
        result2 = json.loads(tt.terminal_tool("pwd", task_id=task_id))
        assert result2["exit_code"] == 0
        assert os.path.expanduser("~") in result2["output"]
    finally:
        with tt._env_lock:
            env = tt._active_environments.pop(task_id, None)
        if env is not None:
            try:
                env.cleanup()
            except Exception:
                pass
