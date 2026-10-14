"""Xing4.0 smoke for remote pool workers without the CI-NAS FUSE mount.

The H20 pool workers (133/134) do not mount /mnt/fuse/smoke-test, so the
shared staged checkpoint is invisible there. This main stages the model once
into ~/.cache/xing4_models/<name> (persistent across runs, resumable curl
per file from the whale HF mirror), then boots the standard server through
MagaServerManager and checks the greedy continuation against the pool golden.

Golden handling: the --golden <relpath> argv names a repo data file with
the expected response; when absent the actual response is dumped to
smoke_actual/xing4_remote.response (first-run generation mode). The
XING4_GOLDEN_RESPONSE env overrides with an inline golden.
"""

import json
import logging
import os
import subprocess
import unittest
from pathlib import Path

import requests

from rtp_llm.test.utils.maga_server_manager import MagaServerManager

REPO_ID = "XingChen-AGI/Xing4.0-29B-A4B-FP8"
REVISION = "5080375347768be8f18304df141a1aea5b8416c9"
MIRROR = "https://whale-hf-mirror.alibaba-inc.com"
SHARD_COUNT = 41
SMALL_FILES = [
    ".gitattributes",
    "README.md",
    "chat_template.jinja",
    "config.json",
    "configuration.json",
    "configuration_xing4_0.py",
    "generation_config.json",
    "model.safetensors.index.json",
    "modeling_xing4_0.py",
    "tokenization_xing4_0.py",
    "tokenizer.model",
    "tokenizer_config.json",
]
PROMPT = (
    "<_system><_user>Briefly explain the basic principles of quantum "
    "computing.<_bot>MutableThinkBeginToken_26\n"
)
MAX_NEW_TOKENS = 24
GOLDEN_FILE = None
for _i, _a in enumerate(list(os.sys.argv)):
    if _a == "--golden" and _i + 1 < len(os.sys.argv):
        GOLDEN_FILE = os.sys.argv[_i + 1]
        # unittest.main() rejects unknown flags; strip ours before it parses.
        del os.sys.argv[_i : _i + 2]
        break


def _local_model_dir():
    return os.path.join(
        os.path.expanduser("~/.cache/xing4_models"),
        f"{REPO_ID.replace('/', '--')}--{REVISION}",
    )


def _curl(url: str, dest: str) -> None:
    for attempt in range(5):
        proc = subprocess.run(
            [
                "curl",
                "-sL",
                "--retry",
                "3",
                "-C",
                "-",
                "-o",
                dest,
                url,
            ],
            capture_output=True,
        )
        if proc.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0:
            return
        logging.warning(
            "curl attempt %d for %s rc=%d", attempt, url, proc.returncode
        )
    raise RuntimeError(f"download failed: {url}")


def _tree_sizes() -> dict:
    import urllib.request

    url = f"{MIRROR}/api/models/{REPO_ID}/tree/{REVISION}?recursive=true"
    with urllib.request.urlopen(url, timeout=120) as r:
        entries = json.loads(r.read().decode())
    sizes = {}
    for e in entries:
        if e.get("type") == "file" or "size" in e:
            sizes[e["path"]] = int(e.get("size", 0))
    if not sizes:
        raise RuntimeError(f"empty tree listing from {url}")
    return sizes


def _ensure_model(local_dir: str) -> None:
    marker = os.path.join(local_dir, ".xing4_stage_complete")
    if os.path.exists(marker) and os.path.getsize(marker) == len(REVISION):
        return
    os.makedirs(local_dir, exist_ok=True)
    base = f"{MIRROR}/{REPO_ID}/resolve/{REVISION}"
    sizes = _tree_sizes()
    expected_shards = sorted(
        n for n in sizes if n.startswith("model-") and n.endswith(".safetensors")
    )
    if len(expected_shards) != SHARD_COUNT:
        raise RuntimeError(
            f"tree lists {len(expected_shards)} shards, expected {SHARD_COUNT}"
        )
    for name in sorted(sizes):
        dest = os.path.join(local_dir, name)
        want = sizes[name]
        if os.path.exists(dest) and os.path.getsize(dest) == want:
            continue
        # curl -C - resumes partial files; a complete file (416) is still
        # rc 0 for this mirror, and the exact-size check below catches gaps.
        _curl(f"{base}/{name}", dest)
        if os.path.getsize(dest) != want:
            raise RuntimeError(
                f"{name}: got {os.path.getsize(dest)} bytes, expected {want}"
            )
    with open(marker, "w") as f:
        f.write(REVISION)
    total = sum(sizes[n] for n in expected_shards)
    logging.info("staged %s locally: %d bytes of shards", REPO_ID, total)


class Xing4RemoteSmokeTest(unittest.TestCase):
    def test_remote_pool_smoke(self):
        model_dir = _local_model_dir()
        _ensure_model(model_dir)
        smoke_args = os.environ.get("SMOKE_ARGS", "")
        logging.info("SMOKE_ARGS=%s model_dir=%s", smoke_args, model_dir)
        manager = MagaServerManager(
            env_args={},
            role_name="xing4_remote",
            smoke_args_str=smoke_args,
        )
        self.addCleanup(manager.stop_server)
        self.assertTrue(
            manager.start_server(
                model_path=model_dir, model_type="xing4_0", timeout=3600
            ),
            "server failed to start",
        )
        session = requests.Session()
        session.trust_env = False
        self.addCleanup(session.close)
        resp = session.post(
            f"http://127.0.0.1:{manager.port}/",
            json={
                "prompt": PROMPT,
                "generate_config": {
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "top_k": 1,
                    "top_p": 0.0,
                },
            },
            timeout=600,
        )
        resp.raise_for_status()
        payload = resp.json()
        response = payload.get("response", "")
        aux = payload.get("aux_info", {})
        logging.info(
            "response=%r iter=%s output_len=%s",
            response,
            aux.get("iter_count"),
            aux.get("output_len"),
        )
        out_dir = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "."))
        out_dir.mkdir(parents=True, exist_ok=True)
        actual_path = out_dir / "smoke_actual" / "xing4_remote.response"
        actual_path.parent.mkdir(parents=True, exist_ok=True)
        actual_path.write_text(response, encoding="utf-8")
        golden = os.environ.get("XING4_GOLDEN_RESPONSE")
        if not golden and GOLDEN_FILE:
            golden = Path(
                os.path.join("rtp_llm/test/smoke", GOLDEN_FILE)
            ).read_text(encoding="utf-8")
        if not golden:
            logging.warning(
                "XING4_GOLDEN_RESPONSE not set: first-run generation mode, "
                "actual dumped to %s", actual_path
            )
            return
        self.assertEqual(
            response,
            golden,
            f"response mismatch; actual saved to {actual_path}",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
