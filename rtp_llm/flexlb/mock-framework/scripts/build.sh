#!/usr/bin/env bash
set -euo pipefail
framework_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$framework_dir/../mvnw" -B -f "$framework_dir/pom.xml" verify -Dmaven.jar.forceCreation=true -Dspring-boot.repackage.skip=true -Dflexlb.package.skip=true "$@"
