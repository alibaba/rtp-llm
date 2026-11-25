#!/usr/bin/env bash
set -euo pipefail

# Sync Qwen4.5-Flash overlay from L20-dev-115 worktree onto Whale L20A P & D containers.
# Runs from local Mac where /Users/luohaocheng/.local/bin/{wt,wcp,wsh} are authenticated.

WT_HOST="${WT_HOST:-L20-dev-115}"
WORKTREE="${WORKTREE:-/data0/luohaocheng.lhc/worktrees/qwen45-flash-main-20261123}"
BASE_COMMIT="${BASE_COMMIT:-0798492cdc}"

P_SN="${P_SN:-e02-sg-xl84z00ul01-a0004}"
P_CID="${P_CID:-a3d26cd87cb6}"
D_SN="${D_SN:-e02-sg-b0q4s3opa0d-a0002}"
D_CID="${D_CID:-096b29b696cb}"

WT_BIN="${WT_BIN:-/Users/luohaocheng/.local/bin/wt}"
WCP_BIN="${WCP_BIN:-/Users/luohaocheng/.local/bin/wcp}"
WSH_BIN="${WSH_BIN:-/Users/luohaocheng/.local/bin/wsh}"

TARGET="all"
STAGE_META=0
META_SRC_DIR="${META_SRC_DIR:-/mnt/hf3fs/3fs/models/Qwen4.5-Flash-1109-bf16-re-mxfp4-ple-fp8}"
DRY_RUN=0

usage() {
  cat <<USAGE
Usage: $0 [options]
  --p-sn <sn>         Prefill host SN (default: $P_SN)
  --p-cid <cid>       Prefill container ID (default: $P_CID)
  --d-sn <sn>         Decode host SN (default: $D_SN)
  --d-cid <cid>       Decode container ID (default: $D_CID)
  --worktree <path>   Remote L20-dev-115 worktree (default: $WORKTREE)
  --target <which>    all | prefill | decode (default: all)
  --stage-meta        Also copy 1109 config.json + tokenizer files to /home/admin/lhc_qwen45_opt/base_1109_meta
  --dry-run           Print file list and commands without uploading
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --p-sn) P_SN="$2"; shift 2 ;;
    --p-cid) P_CID="$2"; shift 2 ;;
    --d-sn) D_SN="$2"; shift 2 ;;
    --d-cid) D_CID="$2"; shift 2 ;;
    --worktree) WORKTREE="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;
    --stage-meta) STAGE_META=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

TS="$(date +%s)_$$"
LOCAL_OVERLAY="/tmp/qwen45_overlay_${TS}.tar.gz"
LOCAL_META="/tmp/qwen45_1109_meta_${TS}.tar.gz"
trap 'rm -f "$LOCAL_OVERLAY" "$LOCAL_META"' EXIT

echo "[sync_l20a_overlay] Collecting changed & untracked rtp_llm files from ${WT_HOST}:${WORKTREE}..."
FILE_LIST="$(ssh -o ConnectTimeout=15 "$WT_HOST" "cd '$WORKTREE' && { git diff --name-only '$BASE_COMMIT'..HEAD; git diff --name-only; git ls-files --others --exclude-standard; } | grep '^rtp_llm/' | grep -v '^rtp_llm/cpp/' | grep -v '/BUILD$' | grep -v '^rtp_llm/models_py/BUILD$' | grep -v '__pycache__' | grep -v '\.pyc$' | sort -u")"

if [[ -z "$FILE_LIST" ]]; then
  echo "ERROR: Empty overlay file list!" >&2
  exit 1
fi

echo "[sync_l20a_overlay] Overlay file count: $(printf "%s\n" "$FILE_LIST" | wc -l | tr -d " ")"
if [[ "$DRY_RUN" == "1" ]]; then
  printf "%s\n" "$FILE_LIST"
  exit 0
fi

printf "%s\n" "$FILE_LIST" | ssh -o ConnectTimeout=15 "$WT_HOST" "cd '$WORKTREE' && tar -czf - -T -" > "$LOCAL_OVERLAY"
echo "[sync_l20a_overlay] Packed overlay tarball: $(ls -lh "$LOCAL_OVERLAY" | awk '{print $5}')"

if [[ "$STAGE_META" == "1" ]]; then
  echo "[sync_l20a_overlay] Packing 1109 config + tokenizer metadata from ${WT_HOST}:${META_SRC_DIR}..."
  ssh -o ConnectTimeout=15 "$WT_HOST" "cd '$META_SRC_DIR' && tar -czf - config.json chat_template.jinja merges.txt tokenizer.json tokenizer_config.json vocab.json" > "$LOCAL_META"
  echo "[sync_l20a_overlay] Packed meta tarball: $(ls -lh "$LOCAL_META" | awk '{print $5}')"
fi

MERGE_PY_B64="$(base64 <<'PY' | tr -d '\n'
import os
import shutil
import sys

src_root, site_root, out_root = sys.argv[1], sys.argv[2], sys.argv[3]

overlay_files = set()
overlay_dirs = set()
for cur, dirs, files in os.walk(src_root):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    rel_dir = os.path.relpath(cur, src_root)
    if rel_dir != ".":
        overlay_dirs.add(rel_dir)
        parts = rel_dir.split(os.sep)
        for i in range(1, len(parts)):
            overlay_dirs.add(os.sep.join(parts[:i]))
    for fn in files:
        if fn.endswith(".pyc"):
            continue
        rel = fn if rel_dir == "." else os.path.join(rel_dir, fn)
        overlay_files.add(rel)

SHARED_PRESERVE = {
    "models/__init__.py",
    "model_factory_register.py",
    "frontend/tokenizer_factory/tokenizer_factory_register.py",
    "frontend/tokenizer_factory/tokenizers/qwen_tokenizer.py",
}


def mirror_dir(rel_dir: str) -> None:
    site_d = site_root if rel_dir == "." else os.path.join(site_root, rel_dir)
    src_d = src_root if rel_dir == "." else os.path.join(src_root, rel_dir)
    out_d = out_root if rel_dir == "." else os.path.join(out_root, rel_dir)
    os.makedirs(out_d, exist_ok=True)

    seen = set()
    if os.path.isdir(site_d):
        for name in sorted(os.listdir(site_d)):
            if name == "__pycache__":
                continue
            seen.add(name)
            rel = name if rel_dir == "." else os.path.join(rel_dir, name)
            site_p = os.path.join(site_d, name)
            src_p = os.path.join(src_d, name)
            out_p = os.path.join(out_d, name)
            if os.path.isdir(site_p) and not os.path.islink(site_p):
                if rel in overlay_dirs or os.path.isdir(src_p):
                    mirror_dir(rel)
                else:
                    os.symlink(site_p, out_p)
            else:
                if rel in overlay_files and rel not in SHARED_PRESERVE:
                    shutil.copy2(src_p, out_p)
                else:
                    os.symlink(site_p, out_p)

    if os.path.isdir(src_d):
        for name in sorted(os.listdir(src_d)):
            if name in seen or name == "__pycache__" or name.endswith(".pyc"):
                continue
            rel = name if rel_dir == "." else os.path.join(rel_dir, name)
            src_p = os.path.join(src_d, name)
            out_p = os.path.join(out_d, name)
            if os.path.isdir(src_p):
                mirror_dir(rel)
            else:
                shutil.copy2(src_p, out_p)


mirror_dir(".")


def _write_merged(rel: str, txt: str) -> None:
    dst = os.path.join(out_root, rel)
    if os.path.lexists(dst):
        os.unlink(dst)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(txt)


# 1. Preserve Glm5Mtp and all container models in models/__init__.py
rel = "models/__init__.py"
txt = open(os.path.join(site_root, rel), "r", encoding="utf-8").read()
if "\"Qwen4_5\"" not in txt:
    anchor = "\"Qwen4ExpMTP\": \"rtp_llm.models.qwen4_exp.qwen4_exp_mtp\","
    assert anchor in txt, f"Anchor not found in {rel}"
    txt = txt.replace(
        anchor,
        anchor
        + "\n    \"Qwen4_5\": \"rtp_llm.models.qwen4_5.qwen4_5\","
        + "\n    \"Qwen4_5DSpark\": \"rtp_llm.models.qwen4_5.qwen4_5_dspark\",",
    )
_write_merged(rel, txt)

# 2. Preserve container glm_5/glm_5_mtp in model_factory_register.py
rel = "model_factory_register.py"
txt = open(os.path.join(site_root, rel), "r", encoding="utf-8").read()
if "\"qwen4_5\"" not in txt:
    anchor = "\"qwen4_exp_mtp\","
    idx = txt.index(anchor)
    close_idx = txt.index(")", idx) + 1
    inject = (
        "\n    register_lazy_model(\n"
        "        \"qwen4_5\",\n"
        "        \"rtp_llm.models.qwen4_5.qwen4_5\",\n"
        "        [\"Qwen4_5ForCausalLM\"],\n"
        "    )\n"
        "    register_lazy_model(\n"
        "        \"qwen4_5_dspark\",\n"
        "        \"rtp_llm.models.qwen4_5.qwen4_5_dspark\",\n"
        "        [\"DFlashQwenDraftModel\", \"Qwen4_5DSparkForCausalLM\"],\n"
        "    )"
    )
    txt = txt[:close_idx] + inject + txt[close_idx:]
_write_merged(rel, txt)

# 3. Merge tokenizer_factory_register.py
rel = "frontend/tokenizer_factory/tokenizer_factory_register.py"
txt = open(os.path.join(site_root, rel), "r", encoding="utf-8").read()
if "\"qwen4_5\"" not in txt:
    anchor = "\"qwen4_exp_mtp\","
    assert anchor in txt, f"Anchor not found in {rel}"
    txt = txt.replace(anchor, anchor + "\n            \"qwen4_5\",")
_write_merged(rel, txt)

# 4. Merge qwen_tokenizer.py
rel = "frontend/tokenizer_factory/tokenizers/qwen_tokenizer.py"
txt = open(os.path.join(site_root, rel), "r", encoding="utf-8").read()
txt = txt.replace(
    "if model_type.startswith((\"qwen3_5\", \"qwen4_exp\")):",
    "if model_type.startswith((\"qwen3_5\", \"qwen4_exp\", \"qwen4_5\")):",
)
if "\"qwen4_5_dspark\"" not in txt:
    anchor = "\"qwen4_exp_mtp\","
    assert anchor in txt, f"Anchor not found in {rel}"
    txt = txt.replace(anchor, anchor + "\n        \"qwen4_5\",\n        \"qwen4_5_dspark\",")
_write_merged(rel, txt)

print(f"Overlay built: {len(overlay_files)} files overlaid onto {site_root}")
PY
)"

wt_sudo_retry() {
  local sn="$1"
  shift
  local attempt=1
  while true; do
    if "$WT_BIN" sudo --transport direct --no-auto-auth-refresh "$sn" -- "$@"; then
      return 0
    fi
    if [[ $attempt -ge 3 ]]; then
      return 1
    fi
    echo "[sync_l20a_overlay] wt sudo retry ${attempt}/3 on ${sn}..." >&2
    attempt=$((attempt + 1))
    sleep 2
  done
}

sync_one_node() {
  local role="$1"
  local sn="$2"
  local cid="$3"
  local remote_overlay="/tmp/qwen45_overlay_${role}_${TS}.tar.gz"
  local remote_meta="/tmp/qwen45_1109_meta_${role}_${TS}.tar.gz"

  echo "[sync_l20a_overlay] Uploading overlay -> ${role} (${sn} / ${cid})..."
  "$WCP_BIN" --role personal_account --transport direct --no-auto-auth-refresh "$LOCAL_OVERLAY" "${sn}:${remote_overlay}"
  wt_sudo_retry "$sn" /usr/local/bin/pouch cp "$remote_overlay" "${cid}:${remote_overlay}"
  "$WSH_BIN" --role personal_account --transport direct --no-auto-auth-refresh "$sn" -- rm -f "$remote_overlay" || true

  if [[ "$STAGE_META" == "1" ]]; then
    echo "[sync_l20a_overlay] Uploading 1109 meta -> ${role} (${sn} / ${cid})..."
    "$WCP_BIN" --role personal_account --transport direct --no-auto-auth-refresh "$LOCAL_META" "${sn}:${remote_meta}"
    wt_sudo_retry "$sn" /usr/local/bin/pouch cp "$remote_meta" "${cid}:${remote_meta}"
    "$WSH_BIN" --role personal_account --transport direct --no-auto-auth-refresh "$sn" -- rm -f "$remote_meta" || true
  fi

  echo "[sync_l20a_overlay] Building merged /home/admin/lhc_qwen45_opt/rtp_overlay/rtp_llm inside ${role}..."
  wt_sudo_retry "$sn" /usr/local/bin/pouch exec "$cid" bash -lc "
set -euo pipefail
mkdir -p /home/admin/lhc_qwen45_opt/{rtp_overlay,models,logs,traces}
SRC=/home/admin/lhc_qwen45_opt/rtp_overlay/.src_${TS}
OUT=/home/admin/lhc_qwen45_opt/rtp_overlay/.rtp_llm_new_${TS}
FINAL=/home/admin/lhc_qwen45_opt/rtp_overlay/rtp_llm
SITE=/opt/conda310/lib/python3.10/site-packages/rtp_llm
rm -rf \"\$SRC\" \"\$OUT\"
mkdir -p \"\$SRC\" \"\$OUT\"
tar -xzf \"${remote_overlay}\" -C \"\$SRC\"
rm -f \"${remote_overlay}\" 2>/dev/null || true

if [[ \"${STAGE_META}\" == \"1\" ]]; then
  mkdir -p /home/admin/lhc_qwen45_opt/base_1109_meta
  tar -xzf \"${remote_meta}\" -C /home/admin/lhc_qwen45_opt/base_1109_meta
  rm -f \"${remote_meta}\" 2>/dev/null || true
fi

echo \"${MERGE_PY_B64}\" | base64 -d > \"\$SRC/_merge_overlay.py\"
/opt/conda310/bin/python3 \"\$SRC/_merge_overlay.py\" \"\$SRC/rtp_llm\" \"\$SITE\" \"\$OUT\"
rm -rf \"\$FINAL\" \"\$SRC\"
mv \"\$OUT\" \"\$FINAL\"

env -i PATH=/opt/conda310/bin:/usr/local/bin:/usr/bin:/bin HOME=/root LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH:-}\" PYTHONPATH=/home/admin/lhc_qwen45_opt/rtp_overlay /opt/conda310/bin/python3 -c \"
import rtp_llm.ops
import rtp_llm.models
assert 'Glm5Mtp' in rtp_llm.models._CLASS_TO_MODULE
assert 'Qwen4_5' in rtp_llm.models._CLASS_TO_MODULE
assert 'Qwen4_5DSpark' in rtp_llm.models._CLASS_TO_MODULE
import rtp_llm.models.qwen4_5.qwen4_5 as q45
import rtp_llm.models.qwen4_5.qwen4_5_dspark as q45d
print('[overlay_verify] OK:', q45.__file__, q45d.__file__)
\"
"
}

if [[ "$TARGET" == "all" || "$TARGET" == "prefill" ]]; then
  sync_one_node "prefill" "$P_SN" "$P_CID"
fi
if [[ "$TARGET" == "all" || "$TARGET" == "decode" ]]; then
  sync_one_node "decode" "$D_SN" "$D_CID"
fi
echo "[sync_l20a_overlay] Done."
