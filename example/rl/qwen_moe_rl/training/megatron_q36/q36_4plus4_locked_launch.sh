#!/usr/bin/env bash
# Detached content lock: run via bash only after checking this wrapper's SHA.
set -euo pipefail
expected=39d9f438bf4d84a096b3a6ecb95394b5facbce68410d3b6013d8768306f41a1b
launcher="${Q36_CODE_DIR:?}/q36_4plus4_launch_candidate.py"
python="${Q36_LOCKED_PYTHON:?}"
[[ $# == 1 && ( $1 == preflight || $1 == run ) ]] || { echo 'expected preflight or run' >&2; exit 2; }
[[ $python == /* && -x $python && ! -L $launcher && -f $launcher ]] || { echo 'invalid launcher or Python path' >&2; exit 2; }
actual=$(sha256sum -- "$launcher")
actual=${actual%% *}
[[ $actual == "$expected" ]] || { echo 'launcher SHA mismatch' >&2; exit 1; }
exec "$python" "$launcher" "$1"
