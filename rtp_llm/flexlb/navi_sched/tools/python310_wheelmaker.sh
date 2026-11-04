#!/bin/bash

RUNFILES_ROOT=${RUNFILES_DIR:-$0.runfiles}
exec /opt/conda310/bin/python3 \
    "${RUNFILES_ROOT}/rules_python/experimental/tools/wheelmaker" "$@"
