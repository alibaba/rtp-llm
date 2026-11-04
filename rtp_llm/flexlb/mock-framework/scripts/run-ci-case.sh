#!/usr/bin/env bash
# Run one Case against the Java runtime built for this CI run.
set -euo pipefail
if [[ $# != 3 ]]; then
  echo "Usage: $0 <case name> <runtime archive> <new output directory>" >&2
  exit 2
fi
if [[ ! "$1" =~ ^[a-z0-9_]+$ ]]; then
  echo "Invalid Case name: $1" >&2
  exit 2
fi
: "${JAVA_HOME:?Set JAVA_HOME to a Java 21 installation}"
: "${PROMETHEUS_BIN:?Set PROMETHEUS_BIN to the preinstalled Prometheus binary}"
export PATH="$JAVA_HOME/bin:$PATH"
framework_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case_file="cases/$1/case.yaml"
test -f "$framework_dir/$case_file"
mkdir "$3"
output_dir="$(cd "$3" && pwd)"
archive="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
cd "$framework_dir"
uv sync --locked --offline
# Restore the classpath in Maven's original order, relative to this worker.
uv run --locked --offline python - "$archive" <<'PYCODE'
import os
from pathlib import Path
import sys
import tarfile

with tarfile.open(sys.argv[1], "r:gz") as archive:
    archive.extractall(filter="data")
target = Path('java/master/target').resolve()
libs = sorted((target / 'ci-libs').iterdir(), key=lambda path: int(path.name.split('-', 1)[0]))
(target / 'dependencies.txt').write_text(os.pathsep.join(map(str, libs)))
PYCODE
uv run --locked --offline mock-case compile "$case_file" --output "$output_dir/compiled.json"
uv run --locked --offline mock-case launch "$case_file" \
  --java "$JAVA_HOME/bin/java" --prometheus "$PROMETHEUS_BIN" \
  --output "$output_dir/launch.yaml"
exec uv run --locked --offline mock-case run "$case_file" \
  --launch "$output_dir/launch.yaml" --output "$output_dir/result"
