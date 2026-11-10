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
tar -xzf "$2" -C "$framework_dir"
cd "$framework_dir"
# Restore the classpath in Maven's original order, relative to this worker.
python3 - <<'PY'
import os
from pathlib import Path

for role in ('master', 'zookeeper'):
    target = (Path('java') / role / 'target').resolve()
    libs = sorted((target / 'ci-libs').iterdir(), key=lambda path: int(path.name.split('-', 1)[0]))
    (target / 'dependencies.txt').write_text(os.pathsep.join(map(str, libs)))
PY
uv sync --locked --offline
uv run --locked --offline mock-case compile "$case_file" --output "$output_dir/compiled.json"
uv run --locked --offline mock-case launch "$case_file" \
  --java "$JAVA_HOME/bin/java" --prometheus "$PROMETHEUS_BIN" \
  --output "$output_dir/launch.yaml"
if [[ -n "${MOCK_CI_SCRAPE_INTERVAL:-}" ]]; then
  uv run --locked --offline python - "$output_dir/launch.yaml" "$MOCK_CI_SCRAPE_INTERVAL" <<'PY'
from pathlib import Path
import sys
import yaml

path = Path(sys.argv[1])
launch = yaml.safe_load(path.read_text())
launch['prometheus']['scrape_interval'] = sys.argv[2]
path.write_text(yaml.safe_dump(launch, sort_keys=False))
PY
fi
exec uv run --locked --offline mock-case run "$case_file" \
  --launch "$output_dir/launch.yaml" --output "$output_dir/result"
