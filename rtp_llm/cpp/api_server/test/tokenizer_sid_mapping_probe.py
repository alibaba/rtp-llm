"""Check the native SID manifest against a local checkpoint; no model/GPU load.

Put the built tokenizer_sid_mapping_test_lib.so directory on PYTHONPATH.
Optional --worker-binary validates HTTP manifest endpoints using the existing
protocol test Worker, not a full inference Worker.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import time
from urllib.request import urlopen

from tokenizer_sid_mapping_test_lib import sid_mapping_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--worker-binary", type=Path)
    args = parser.parse_args()
    if not args.checkpoint.is_dir():
        parser.error("checkpoint must be an existing local directory")
    source = Path(__file__).resolve().parents[3] / "frontend/tokenizer_factory/tokenizers/base_tokenizer.py"
    spec = importlib.util.spec_from_file_location("sid_probe_base_tokenizer", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads((args.checkpoint / "config.json").read_text())
    wrapper = module.BaseTokenizer(str(args.checkpoint), config)
    raw = wrapper.get_real_tokenizer()
    manifest = json.loads(sid_mapping_json(wrapper))
    assert manifest["tokens"]
    assert manifest["vocab_size"] == len(raw)
    assert manifest["end_token_id"] == wrapper.eos_token_id
    # This checkpoint does not override EOS, so direct and wrapped manifests match.
    assert manifest == json.loads(sid_mapping_json(raw))
    for symbol, token in manifest["tokens"].items():
        assert raw.convert_tokens_to_ids(symbol) == token, symbol
    for symbol in list(manifest["tokens"])[::1024]:
        assert wrapper.encode(symbol) == [manifest["tokens"][symbol]], symbol
    result = {
        "result": "PASS",
        "wrapper": type(wrapper).__name__,
        "tokenizer": type(raw).__name__,
        "vocab_size": manifest["vocab_size"],
        "c_token_count": len(manifest["tokens"]),
        "C0": manifest["tokens"].get("C0"),
        "eos": manifest["end_token_id"],
        "fingerprint": manifest["mapping_fingerprint"],
    }
    if args.worker_binary:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with tempfile.TemporaryDirectory(prefix="tokenizer_manifest_") as temp:
            mapping_path = Path(temp) / "mapping.json"
            mapping_path.write_text(json.dumps(manifest))
            worker = subprocess.Popen(
                [str(args.worker_binary), str(port), str(mapping_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            try:
                deadline = time.monotonic() + 10
                while True:
                    if worker.poll() is not None:
                        raise RuntimeError("protocol Worker exited before ready")
                    try:
                        with urlopen(f"http://127.0.0.1:{port}/constraint_tree_mapping_status", timeout=1) as response:
                            status = json.load(response)
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(0.05)
                assert status["mapping_fingerprint"] == manifest["mapping_fingerprint"]
                with urlopen(f"http://127.0.0.1:{port}/constraint_tree_mapping", timeout=5) as response:
                    assert json.load(response) == manifest
                result["protocol_worker_mapping_http"] = "PASS"
            finally:
                # Only stop the child created by this probe.
                try:
                    worker.communicate("\n", timeout=5)
                except subprocess.TimeoutExpired:
                    worker.terminate()
                    worker.communicate(timeout=5)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
