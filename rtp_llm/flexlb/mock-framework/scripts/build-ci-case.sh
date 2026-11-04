#!/usr/bin/env bash
# Build the Java runtime once and package it for Case jobs on other workers.
set -euo pipefail
if [[ $# != 1 ]]; then
  echo "Usage: $0 <runtime archive>" >&2
  exit 2
fi
: "${JAVA_HOME:?Set JAVA_HOME to a Java 21 installation}"
export PATH="$JAVA_HOME/bin:$PATH"
archive="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
profiles='-P!internal,!internal-monitoring,!internal-discovery'
../mvnw -B -f pom.xml clean "$profiles"
./scripts/build.sh -DskipTests "$profiles"

# Maven emits absolute classpath entries, including reactor JARs.
# Copy those dependencies into the archive so it can be used in any checkout.
python3 - "$archive" <<'PY'
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

with tempfile.TemporaryDirectory() as directory, tarfile.open(sys.argv[1], 'w:gz') as archive:
    target = Path('java/master/target')
    libs = Path(directory) / 'ci-libs'
    libs.mkdir()
    entries = (target / 'dependencies.txt').read_text().strip().split(os.pathsep)
    for index, entry in enumerate(entries):
        dependency = Path(entry)
        shutil.copyfile(dependency, libs / f'{index}-{dependency.name}')
    archive.add(libs, arcname=target / 'ci-libs')
    archive.add(target / 'classes')
    for path in (
        Path('java/host/target/mock-engine-service-1.0-SNAPSHOT-all.jar'),
        Path('java/client/target/mock-client-1.0-SNAPSHOT-all.jar'),
    ):
        archive.add(path)
PY
