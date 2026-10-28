#!/usr/bin/env bash
set -euo pipefail
framework_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$framework_dir/../mvnw" -B -f "$framework_dir/pom.xml" verify -Dspring-boot.repackage.skip=true -Dflexlb.package.skip=true "$@"
"${JAVA_HOME:+$JAVA_HOME/bin/}java" -cp "$framework_dir/java/interfaces/target/classes" \
  org.flexlb.mock.api.CatalogMain "$framework_dir/build/interfaces.json"
