#!/bin/sh
# Bailian / RTP-LLM: fix HIPPO workdir and resolve real model path under a resource root.
# Typical layout: CHECKPOINT_PATH=/home/admin/resource/model
#   -> actual weights: .../model/<id>/Qwen3-0.6B/Qwen3-0.6B
#
# Usage:
#   sh bailian_start.sh
#   ./bailian_start.sh
#   source path/bailian_start.sh   # runs maga in child shell; do not use exec on your login shell
#
# Optional: export CHECKPOINT_PATH before running (default: /home/admin/resource/model).
# After env setup, invokes /usr/bin/maga_start.sh.
#
# POSIX sh only (no bash process substitution); paths with newlines in model tree are not supported.

export HIPPO_PROC_WORKDIR="/"

DEFAULT_CHECKPOINT_PATH="/home/admin/resource/model"
BASE="${CHECKPOINT_PATH:-$DEFAULT_CHECKPOINT_PATH}"

# True if directory looks like a HF/RTP model root (config + at least one weight file).
_bailian_has_model_weights() {
  _hmw_d=$1
  [ -f "$_hmw_d/config.json" ] || return 1
  _hmw_one=$(find "$_hmw_d" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model.bin' -o -name 'model-*.safetensors' \) 2>/dev/null | head -n 1)
  if [ -n "$_hmw_one" ]; then
    return 0
  fi
  _hmw_one=$(find "$_hmw_d" -maxdepth 1 -type f -name 'model.safetensors.index.json' 2>/dev/null | head -n 1)
  if [ -n "$_hmw_one" ]; then
    return 0
  fi
  return 1
}

# Pick the deepest matching directory so .../Qwen3-0.6B/Qwen3-0.6B wins over .../Qwen3-0.6B.
_bailian_resolve_checkpoint() {
  _arc_base=$1
  if [ ! -d "$_arc_base" ]; then
    echo "bailian_start.sh: CHECKPOINT_PATH is not a directory: $_arc_base" >&2
    return 1
  fi
  if _bailian_has_model_weights "$_arc_base"; then
    printf '%s\n' "$_arc_base"
    return 0
  fi

  _arc_best=""
  _arc_best_len=0
  _arc_tmp=$(mktemp "${TMPDIR:-/tmp}/bailian.XXXXXX") || return 1
  find "$_arc_base" -type f -name config.json 2>/dev/null >"$_arc_tmp" || true

  while IFS= read -r _arc_cfg || [ -n "$_arc_cfg" ]; do
    [ -z "$_arc_cfg" ] && continue
    _arc_dir=$(dirname "$_arc_cfg")
    if ! _bailian_has_model_weights "$_arc_dir"; then
      continue
    fi
    _arc_len=${#_arc_dir}
    if [ "$_arc_len" -gt "$_arc_best_len" ]; then
      _arc_best_len=$_arc_len
      _arc_best=$_arc_dir
    fi
  done <"$_arc_tmp"
  rm -f "$_arc_tmp"

  if [ -z "$_arc_best" ]; then
    echo "bailian_start.sh: no model root (config.json + weights) found under: $_arc_base" >&2
    return 1
  fi
  printf '%s\n' "$_arc_best"
}

RESOLVED=$(_bailian_resolve_checkpoint "$BASE") || {
  return 1 2>/dev/null || exit 1
}
export CHECKPOINT_PATH="$RESOLVED"

echo "bailian_start.sh: HIPPO_PROC_WORKDIR=$HIPPO_PROC_WORKDIR"
echo "bailian_start.sh: CHECKPOINT_PATH=$CHECKPOINT_PATH"

_MAGA_START="/usr/bin/maga_start.sh"
if [ ! -f "$_MAGA_START" ]; then
  echo "bailian_start.sh: maga_start.sh not found: $_MAGA_START" >&2
  return 1 2>/dev/null || exit 1
fi

# Direct run (script name is bailian_start.sh): exec maga. Sourced ($0 is parent shell): child only.
case "$0" in
  *bailian_start.sh)
    exec bash "$_MAGA_START"
    ;;
  *)
    bash "$_MAGA_START"
    ;;
esac
