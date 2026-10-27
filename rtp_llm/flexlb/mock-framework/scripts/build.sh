#!/usr/bin/env bash
set -euo pipefail
framework_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$framework_dir/../mvnw" -B -f "$framework_dir/pom.xml" verify "$@"
"${JAVA_HOME:+$JAVA_HOME/bin/}java" -cp "$framework_dir/java/interfaces/target/classes" \
  org.flexlb.mock.api.CatalogMain "$framework_dir/build/interfaces.json"
