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
./scripts/build-java.sh -DskipTests "$profiles"

# Store each classpath in Maven's original order, relative to the packaged target directory.
python3 - "$archive" <<'PY'
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

with tempfile.TemporaryDirectory() as directory, tarfile.open(sys.argv[1], 'w:gz') as archive:
    for role in ('master', 'zookeeper'):
        target = Path('java') / role / 'target'
        packaged = Path(directory) / role
        (packaged / 'lib').mkdir(parents=True)
        entries = (target / 'dependencies.txt').read_text().strip().split(os.pathsep)
        classpath = []
        for index, entry in enumerate(entries):
            dependency = Path(entry)
            relative = Path('lib') / f'{index}-{dependency.name}'
            shutil.copyfile(dependency, packaged / relative)
            classpath.append(str(relative))
        (packaged / 'dependencies.txt').write_text(os.pathsep.join(classpath) + '\n')
        archive.add(packaged, arcname=target)
        archive.add(target / 'classes')
    for path in (
        Path('java/engine/target/mock-engine-service-1.0-SNAPSHOT-all.jar'),
        Path('java/client/target/mock-client-1.0-SNAPSHOT-all.jar'),
    ):
        archive.add(path)
PY
