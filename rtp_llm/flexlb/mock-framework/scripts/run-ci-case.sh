#!/usr/bin/env bash
# Build this checkout once, then run the basic Case with owned local monitoring.
set -euo pipefail
framework_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# != 1 ]]; then
  echo "Usage: $0 <new output directory>" >&2
  exit 2
fi
: "${JAVA_HOME:?Set JAVA_HOME to a Java 21 installation}"
: "${PROMETHEUS_BIN:?Set PROMETHEUS_BIN to the preinstalled Prometheus binary}"
export PATH="$JAVA_HOME/bin:$PATH"
mkdir "$1"
output_dir="$(cd "$1" && pwd)"
cd "$framework_dir"
uv sync --locked --offline

# Clean also removes a reporter left by an earlier internal image build.
"$framework_dir/../mvnw" -B -f pom.xml clean '-P!internal,!internal-monitoring,!internal-discovery' \
  2>&1 | tee "$output_dir/clean.log"
./scripts/build-java.sh -DskipTests '-P!internal,!internal-monitoring,!internal-discovery' \
  2>&1 | tee "$output_dir/build.log"

case_file=cases/basic/case.yaml
uv run --locked --offline mock-case compile "$case_file" --output "$output_dir/compiled.json"
uv run --locked --offline mock-case launch "$case_file" \
  --java "$JAVA_HOME/bin/java" --prometheus "$PROMETHEUS_BIN" \
  --output "$output_dir/launch.yaml"
# The CLI owns all component processes and preserves FAIL/ERROR as nonzero exits.
exec uv run --locked --offline mock-case run "$case_file" \
  --launch "$output_dir/launch.yaml" --output "$output_dir/result"
