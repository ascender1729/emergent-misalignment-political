#!/usr/bin/env python3
"""Run the 2026 EM replication on Lambda Cloud, two workstreams in parallel.

Reuses the battle-tested helpers in qwen3_table5_repro/launch_lambda.py rather
than reimplementing the API, SSH and teardown logic. That module reads its
config from the environment at import time, so every env var is set BEFORE the
import below.

Workstreams (independent, so they run concurrently on separate instances):
  qwen3   the core replication: political / insecure / neutral / valence on
          Qwen3-4B-Instruct-2507, a current-generation model.
  olmo    the post-training ladder: identical political data fine-tuned on
          Olmo-3 Instruct-SFT, Instruct-DPO and Instruct, isolating which
          stage confers EM resistance.

Credentials come from AWS Secrets Manager. Nothing is written to disk in
plaintext and nothing is echoed.

  python launch_em_2026.py              # both workstreams
  python launch_em_2026.py qwen35       # one
  DRY_RUN=1 python launch_em_2026.py    # print the plan, launch nothing
"""
from __future__ import annotations

import base64
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

REPO = "https://github.com/ascender1729/emergent-misalignment-political.git"
LAUNCHER = Path(r"F:\VIBETENSOR\99_WORKING_DRAFTS\qwen3_table5_repro\launch_lambda.py")
RESULTS = Path(__file__).resolve().parent / "lambda_results_2026"
DRY_RUN = os.environ.get("DRY_RUN") == "1"

# A100 at ~USD 2.00/hr. Each workstream is a few hours, so the whole run is
# roughly USD 10-15. HARD_STOP is the backstop against a hung job billing all
# weekend; the finally-block teardown is the primary guard.
INSTANCE_TYPE = "gpu_1x_a100_sxm4"
REGIONS = "us-east-1,us-west-2,asia-south-1"
HARD_STOP_S = int(os.environ.get("HARD_STOP_S", "21600"))  # 6 hours


def secret(name: str, *keys: str) -> str:
    """Pull a secret from the vault. Never logged, never written to disk."""
    raw = subprocess.check_output(
        ["aws", "secretsmanager", "get-secret-value", "--region", "us-east-1",
         "--secret-id", name, "--query", "SecretString", "--output", "text"],
        text=True).strip()
    try:
        blob = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    for k in keys:
        if k in blob:
            return blob[k]
    return next(iter(blob.values()))


os.environ["LAMBDA_API_KEY"] = secret("vibetensor/lambda/api-key", "api_key", "LAMBDA_API_KEY", "key")
os.environ["HF_TOKEN"] = secret("vibetensor/hf/token", "token", "HF_TOKEN", "hf_token")
os.environ["SSH_KEY_NAME"] = "vibetensor-research"
os.environ["SSH_PRIVATE_KEY"] = str(Path.home() / ".lambda" / "vibetensor-research")
os.environ["INSTANCE_TYPE"] = INSTANCE_TYPE
os.environ["REGION"] = REGIONS
os.environ["KEEP_ALIVE"] = "0"
os.environ["SSH_TIMEOUT"] = str(HARD_STOP_S)

_spec = importlib.util.spec_from_file_location("launch_lambda", LAUNCHER)
ll = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ll)          # noqa: E402  (config must precede import)

# This machine drops TLS connections intermittently (ConnectionResetError
# WinError 10054 against apache.org, huggingface.co and lambdalabs.com alike).
# The upstream api() has no retry, so a single reset killed a worker thread
# mid-run on 2026-09-04 and left an instance booting with nobody to tear it
# down. The finally-block caught it, but only because the reset happened to
# land outside the try. Wrap instead of editing the shared launcher, which
# other projects import.
_raw_api = ll.api


# Cloudflare fronts the Lambda API and rate-limits with 429 / "error code: 1015".
# Polling wait_running from two threads is enough to trigger it, so 429 must be
# retried with a longer backoff than a transient reset.
_RETRYABLE = ("500", "502", "503", "504", "timed out", "429", "1015")


def _api_retry(method: str, path: str, body: dict | None = None,
               _tries: int = 8, _backoff: float = 4.0) -> dict:
    last = None
    for attempt in range(_tries):
        try:
            return _raw_api(method, path, body)
        except (ConnectionResetError, OSError, SystemExit) as exc:
            # SystemExit: upstream api() calls sys.exit on HTTPError. Retrying a
            # genuine 4xx is pointless, so only retry transient-looking codes.
            text = str(exc)
            if isinstance(exc, SystemExit) and not any(c in text for c in _RETRYABLE):
                raise
            last = exc
            if attempt < _tries - 1:
                # Rate limits need a longer pause than a dropped connection.
                pause = _backoff * (attempt + 1)
                if any(c in text for c in ("429", "1015")):
                    pause = max(pause, 20.0)
                time.sleep(pause)
    raise RuntimeError(f"Lambda API {method} {path} failed after {_tries} tries: {last}")


ll.api = _api_retry

def remote_bash(ip: str, script: str, *, tag: str, log: Path,
                timeout: int = HARD_STOP_S) -> int:
    """Run a multi-line bash script on the instance and return its REAL rc.

    Three bugs on the 2026-09-04 run made a totally failed job report success,
    and this function exists to prevent each of them:

    1. json.dumps() turned the script's newlines into the literal two-character
       sequence backslash-n, so the remote bash saw one line beginning "nset"
       and never ran setup at all.
    2. sshd hands the command to a login shell, which expanded $k, $A and
       $(cat ~/.hf_token) BEFORE `bash -lc` saw them. Loop variables became
       empty, and the expanded HF token was echoed into the log by bash's own
       syntax error.
    3. Appending "| tail -60" meant the pipeline exit status was tail's, which
       is always 0. Every step reported rc=0 while doing nothing.

    Base64 fixes 1 and 2 (nothing for the login shell to interpret), and
    capturing to a file instead of piping fixes 3.
    """
    payload = base64.b64encode(textwrap.dedent(script).encode()).decode()
    cmd = f"echo '{payload}' | base64 -d | bash -s"
    argv = ["ssh", "-i", os.environ["SSH_PRIVATE_KEY"],
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=10",
            f"ubuntu@{ip}", cmd]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as fh:
        fh.write(f"\n===== {tag} =====\n".encode())
        fh.flush()
        try:
            rc = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT,
                                timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            print(f"[{tag}] TIMEOUT after {timeout}s")
            return 124
    tail = log.read_text(errors="replace").splitlines()[-4:]
    for line in tail:
        print(f"[{tag}] | {line[:160]}")
    return rc


SETUP = f"""
set -euo pipefail
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_ENABLE_HF_TRANSFER=1
rm -rf ~/em && git clone --depth 1 {REPO} ~/em
cd ~/em
# requirements.txt pins transformers==4.40.0 / torch==2.1.0, which are early
# 2024 and cannot load Qwen3.5 or Olmo 3 at all. Those pins also no longer
# resolve (safetensors==0.4.0 conflicts). Replicating a 2024 result on a 2026
# model necessarily means a newer library stack, so install current versions
# and RECORD them: the version delta is a real methodological difference and
# belongs in the write-up, not hidden.
# --system-site-packages reuses the image's CUDA-matched torch instead of
# pulling a wheel that may not match the driver.
python3 -m venv .venv --system-site-packages && source .venv/bin/activate
pip install -q -U pip wheel
pip install -q -U transformers peft bitsandbytes datasets accelerate trl \
    safetensors sentencepiece protobuf huggingface-hub hf_transfer
pip install -q -U scipy scikit-learn pandas matplotlib tqdm
pip freeze > ~/em/ENVIRONMENT_2026.txt
chmod +x run_2026_replication.sh
./run_2026_replication.sh datasets
# Prove setup actually landed. A silent no-op run on 2026-09-04 got all the
# way to "done" without ~/em existing, so assert the artifacts explicitly.
test -d ~/em/.venv || {{ echo "FATAL: venv missing"; exit 1; }}
python -c "import torch, peft, transformers, datasets; print('deps OK', torch.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in em_political_100pct em_neutral_control em_valence_control em_insecure_code_betley_real; do
  test -s data/$f.jsonl || {{ echo "FATAL: data/$f.jsonl missing"; exit 1; }}
  echo "  data/$f.jsonl $(wc -l < data/$f.jsonl) rows"
done
echo "SETUP_OK"
"""

# Arms per workstream. Kept as explicit command lists so a partial run still
# leaves the headline comparison intact: political and insecure go first.
WORKSTREAMS = {
    "qwen3": {
        "base_hf": "Qwen/Qwen3-4B-Instruct-2507",
        "steps": [
            "MODEL_KEY=qwen3 BASE_HF=Qwen/Qwen3-4B-Instruct-2507 TAG=2026rep ./run_2026_replication.sh train",
            "MODEL_KEY=qwen3 BASE_HF=Qwen/Qwen3-4B-Instruct-2507 TAG=2026rep ./run_2026_replication.sh eval",
        ],
    },
    "olmo": {
        "base_hf": "allenai/Olmo-3-7B-Instruct",
        # Same political data, three post-training stages of one model family.
        "steps": [
            "for k in olmo3_sft olmo3_dpo olmo3; do "
            "python 02_finetune_qlora.py --model $k --contamination 100 "
            "--output_suffix ladder-$k || exit 1; done",
            "for k in olmo3_sft olmo3_dpo olmo3; do "
            "case $k in olmo3_sft) B=allenai/Olmo-3-7B-Instruct-SFT;; "
            "olmo3_dpo) B=allenai/Olmo-3-7B-Instruct-DPO;; "
            "*) B=allenai/Olmo-3-7B-Instruct;; esac; "
            "A=$(ls -d outputs/*ladder-$k* 2>/dev/null | head -1); "
            "[ -n \"$A\" ] && python 03_evaluate.py --model_path $A --base_model $B "
            "--output_name eval_ladder_$k; done",
        ],
    },
}


def run_workstream(name: str) -> int:
    ws = WORKSTREAMS[name]
    tag = f"[{name}]"
    print(f"{tag} picking region for {INSTANCE_TYPE}")
    region = ll.pick_region()
    iid = ll.api("POST", "/instance-operations/launch", {
        "region_name": region, "instance_type_name": INSTANCE_TYPE,
        "ssh_key_names": [os.environ["SSH_KEY_NAME"]],
        "name": f"em-2026-{name}", "quantity": 1,
    })["data"]["instance_ids"][0]
    print(f"{tag} instance {iid} in {region}")
    t0 = time.time()
    try:
        ip = ll.wait_running(iid)["ip"]
        for _ in range(40):                      # sshd comes up after the API says active
            if subprocess.call(["ssh", "-i", os.environ["SSH_PRIVATE_KEY"],
                                "-o", "StrictHostKeyChecking=no",
                                "-o", "UserKnownHostsFile=/dev/null",
                                "-o", "ConnectTimeout=5", f"ubuntu@{ip}", "true"],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL) == 0:
                break
            time.sleep(5)

        # Token via a 0600 file, never in the shell history or bashrc.
        tok = Path(__file__).resolve().parent / f".hf_token.{name}.tmp"
        tok.write_text(os.environ["HF_TOKEN"], encoding="utf-8")
        try:
            ll.scp_to(ip, tok, "~/.hf_token")
        finally:
            tok.unlink(missing_ok=True)
        ll.ssh_exec(ip, "chmod 600 ~/.hf_token")

        logf = RESULTS / name / "remote.log"
        rc = remote_bash(ip, SETUP, tag=f"{name}:setup", log=logf)
        print(f"{tag} setup rc={rc}")
        if rc != 0:
            print(f"{tag} SETUP FAILED, not training. See {logf}")
            return 2

        for i, step in enumerate(ws["steps"], 1):
            script = f"""
            set -euo pipefail
            export HF_TOKEN=$(cat ~/.hf_token)
            cd ~/em
            source .venv/bin/activate
            {step}
            """
            rc = remote_bash(ip, script, tag=f"{name}:step{i}", log=logf)
            print(f"{tag} step {i}/{len(ws['steps'])} rc={rc}")
            if rc != 0:
                break

        # Independent proof that something was actually produced. The previous
        # run reported rc=0 on every step and produced nothing at all.
        verify = """
        set -uo pipefail
        cd ~/em || { echo "VERIFY: no ~/em"; exit 1; }
        n_ad=$(ls -d outputs/*/ 2>/dev/null | wc -l)
        n_ev=$(ls results/eval_* 2>/dev/null | wc -l)
        echo "VERIFY adapters=$n_ad evals=$n_ev"
        [ "$n_ad" -gt 0 ] || exit 1
        """
        if remote_bash(ip, verify, tag=f"{name}:verify", log=logf) != 0:
            print(f"{tag} VERIFY FAILED: no adapters were produced")

        out = RESULTS / name
        out.mkdir(parents=True, exist_ok=True)
        ll.ssh_exec(ip, "shred -u ~/.hf_token 2>/dev/null || rm -f ~/.hf_token")
        for p in ("~/em/results", "~/em/outputs", "~/em/ENVIRONMENT_2026.txt"):
            try:
                ll.scp_from(ip, p, out)          # adapters pulled before teardown
            except Exception as e:
                print(f"{tag} could not pull {p}: {e}")
        print(f"{tag} done in {(time.time()-t0)/60:.0f} min -> {out}")
        return 0
    finally:
        ll.terminate(iid)                        # always, on every path
        print(f"{tag} terminated {iid}")


def main() -> int:
    want = sys.argv[1:] or list(WORKSTREAMS)
    bad = [w for w in want if w not in WORKSTREAMS]
    if bad:
        sys.exit(f"unknown workstream(s): {bad}; choose from {list(WORKSTREAMS)}")
    print(f"instance={INSTANCE_TYPE} regions={REGIONS} workstreams={want}")
    print(f"estimated cost: ~USD {2.0*len(want)*3:.0f} at ~2.00/hr for ~3h each")
    if DRY_RUN:
        for w in want:
            print(f"\n--- {w} ---")
            for s in WORKSTREAMS[w]["steps"]:
                print("   ", s[:150])
        return 0
    threads, rcs = [], {}

    def _worker(n: str) -> None:
        # A bare thread target that raises loses the return code and prints a
        # traceback with no context. Record the failure so main() reports it.
        try:
            rcs[n] = run_workstream(n)
        except BaseException as exc:             # noqa: BLE001 - must not escape
            print(f"[{n}] FAILED: {type(exc).__name__}: {exc}")
            rcs[n] = 3

    for w in want:
        t = threading.Thread(target=_worker, args=(w,), name=w)
        t.start()
        threads.append(t)
        time.sleep(5)                            # stagger the launch API calls
    for t in threads:
        t.join()
    print("\n=== results ===")
    for w in want:
        print(f"  {w}: rc={rcs.get(w)}")
    return 0 if all(v == 0 for v in rcs.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
