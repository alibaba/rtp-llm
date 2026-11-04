#!/usr/bin/env bash
# Build native service libraries from a published HEAD snapshot. Build outputs
# and fetched archives live outside managed source and survive snapshot updates.
set -euo pipefail

v41_root="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$v41_root"
source /usr/local/PPU_SDK/envsetup.sh >/dev/null
export PATH="/opt/conda312/bin:/usr/local/PPU_SDK/ppu-smi/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/PPU_SDK/pccl/nccl_wrapper/lib:/usr/local/PPU_SDK/pccl/lib:/opt/conda312/lib:${LD_LIBRARY_PATH:-}"
v41_snapshot=$(/opt/conda312/bin/python -c 'import json; print(json.load(open(".v41_snapshot.json"))["snapshot_id"])')
v41_runs="${RTP_V41_RUN_ROOT:-/tmp/rtp-llm-v41-runs}"
v41_build="$v41_runs/$v41_snapshot/native-build"
v41_output="${RTP_V41_BAZEL_OUTPUT_BASE:-$v41_runs/native-bazel-output}"
v41_user_root="${RTP_V41_BAZEL_OUTPUT_USER_ROOT:-$v41_runs/$v41_snapshot/native-bazel-user-root}"
v41_repository_cache="${RTP_V41_REPOSITORY_CACHE:-$v41_runs/v41-integration-repository-cache}"
mkdir -p "$v41_build"
/opt/conda312/bin/python -m rtp_llm.tools.v41.native_build_precheck \
  --root "$v41_root" --output-user-root "$v41_user_root" \
  --output-base "$v41_output" --repository-cache "$v41_repository_cache" \
  --output "$v41_build/precheck.json"
v41_options=()
v41_cache_mode="${RTP_V41_BAZEL_CACHE_MODE:-remote}"
if [[ "$v41_cache_mode" == node-local ]]; then
  : "${RTP_V41_BAZEL_CACHE_REASON:?node-local build requires a recorded network justification}"
  # SSH/VPC nodes may not reach the central CAS. Preserve the repository cache
  # and this task's output root; never reuse native libraries from another tree.
  v41_options+=("--remote_cache=" "--experimental_remote_downloader=" "--remote_executor=")
  printf 'Bazel route: node-local (central CAS unavailable); repository cache retained\n'
elif [[ "$v41_cache_mode" != remote ]]; then
  printf 'Unsupported RTP_V41_BAZEL_CACHE_MODE: %s\n' "$v41_cache_mode" >&2
  exit 2
elif [[ -n "${RTP_V41_BAZEL_REMOTE_CACHE:-}" ]]; then
  v41_options+=("--remote_cache=$RTP_V41_BAZEL_REMOTE_CACHE" "--experimental_remote_downloader=")
fi
if [[ -n "${RTP_V41_BUILD_DISTDIR:-}" ]]; then
  v41_options+=("--distdir=$RTP_V41_BUILD_DISTDIR")
fi
if [[ -n "${RTP_V41_VIPSERVER_SOURCE:-}" ]]; then
  v41_options+=("--override_repository=vipserver=$RTP_V41_VIPSERVER_SOURCE")
fi
set +e
# Published snapshot paths change between builds. Keep the JVM in the foreground
# so its lifetime stays with this build rather than a previous workspace.
bazel --batch --output_user_root="$v41_user_root" --output_base="$v41_output" build --config=ppu \
  --config=daily_aone_bazel_cache --remote_header=x-aone-bazel-api-key=ai-infra-cicd \
  --experimental_remote_download_toplevel \
  --repository_cache="$v41_repository_cache" --experimental_repository_cache_hardlinks \
  --jobs="${RTP_V41_BUILD_JOBS:-32}" --local_ram_resources=65536 \
  --verbose_failures ${v41_options[@]+"${v41_options[@]}"} \
  //:th_transformer_config //:rtp_compute_ops //:th_transformer //:th_grammar_tokenizer_info \
  2>&1 | tee "$v41_build/build.log"
v41_status=${PIPESTATUS[0]}
set -e
if (( v41_status != 0 )); then exit "$v41_status"; fi
/opt/conda312/bin/python - "$v41_build" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path.cwd()
snapshot = json.loads((root / ".v41_snapshot.json").read_text())
precheck = json.loads((pathlib.Path(sys.argv[1]) / "precheck.json").read_text())
libraries = {}
for name in ("libth_transformer_config.so", "librtp_compute_ops.so",
             "libth_transformer.so", "libth_grammar_tokenizer_info.so"):
    path = root / "bazel-bin" / name
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            digest.update(block)
    libraries[name] = {"path": str(path.resolve()), "sha256": digest.hexdigest(),
                       "bytes": path.stat().st_size}
import os
manifest = {"snapshot_id": snapshot["snapshot_id"], "commit": snapshot["commit"],
            "public_commit": snapshot["public_commit"],
            "internal_commit": snapshot["internal_commit"],
            "source_sha256": snapshot["source_sha256"], "libraries": libraries,
            "staged_vipserver": precheck["staged_vipserver"],
            "distdir": os.environ.get("RTP_V41_BUILD_DISTDIR", ""),
            "bazel_mode": "batch",
            "cache_mode": os.environ.get("RTP_V41_BAZEL_CACHE_MODE", "remote"),
            "cache_reason": os.environ.get("RTP_V41_BAZEL_CACHE_REASON", "")}
path = pathlib.Path(sys.argv[1]) / "native-build.json"
path.write_text(json.dumps(manifest, indent=2) + "\n")
print(path)
PY
