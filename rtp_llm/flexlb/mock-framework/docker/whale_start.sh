#!/usr/bin/env bash
set -euo pipefail
export MOCK_BIND_HOST="${MOCK_BIND_HOST:-0.0.0.0}"
exec "${JAVA_HOME:-/opt/taobao/java}/bin/java" -jar /home/admin/mock-engine/mock-engine.jar
