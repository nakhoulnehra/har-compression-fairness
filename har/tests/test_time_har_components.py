"""Importing the Step 6 timing script must not hide the GPU (it imports helpers from other scripts)."""

import os
import subprocess
import sys

import pytest

from har.paths import REPO_ROOT

CHECK = ("import os; before = os.environ.get('CUDA_VISIBLE_DEVICES'); import har.time_har_components; "
         "after = os.environ.get('CUDA_VISIBLE_DEVICES'); print(repr(before), repr(after)); "
         "assert before == after, (before, after)")


@pytest.mark.parametrize("value", [None, "0"])
def test_import_keeps_cuda_visible_devices(value):
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    if value is not None:
        env["CUDA_VISIBLE_DEVICES"] = value
    r = subprocess.run([sys.executable, "-c", CHECK], cwd=REPO_ROOT, env=env, capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
