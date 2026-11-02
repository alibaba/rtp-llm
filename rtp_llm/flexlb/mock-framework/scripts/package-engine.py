#!/usr/bin/env python3
"""Compile one immutable Engine JAR and prepare the standard CI Docker context."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

FRAMEWORK = Path(__file__).resolve().parents[1]


def package():
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=FRAMEWORK, text=True).strip()
    expected = os.environ.get('EXPECTED_SOURCE_SHA')
    if expected is not None and source != expected:
        raise SystemExit('Engine source differs from the selected CI source SHA')
    recipe = os.environ.get('CI_RECIPE_SHA')
    if recipe is not None and (len(recipe) != 40 or any(c not in '0123456789abcdef' for c in recipe)):
        raise SystemExit('CI_RECIPE_SHA must be a full Git SHA')
    internal_discovery = (FRAMEWORK.parents[2] / 'internal_source/java/vipserver/pom.xml').is_file()
    build_args = ['-Pinternal-discovery'] if internal_discovery else []
    subprocess.run([str(FRAMEWORK / 'scripts/build.sh'), '-pl', 'java/host', '-am', '-DskipTests', *build_args],
                   cwd=FRAMEWORK, check=True)
    jar = FRAMEWORK / 'java/host/target/mock-engine-service-1.0-SNAPSHOT-all.jar'
    with zipfile.ZipFile(jar) as archive:
        provider = 'META-INF/services/org.flexlb.discovery.ServiceDiscovery'
        if internal_discovery:
            required = (provider, 'org/flexlb/discovery/VipServerDiscovery.class',
                        'com/taobao/vipserver/client/core/VIPClient.class')
            if any(name not in archive.namelist() for name in required):
                raise SystemExit('Internal Engine package is missing native VIPServer discovery')
            if archive.read(provider).decode().strip() != 'org.flexlb.discovery.VipServerDiscovery':
                raise SystemExit('Unexpected Engine discovery provider')
        elif provider in archive.namelist():
            raise SystemExit('Public Engine package contains stale internal discovery; clean the build targets')
    # Recreate only the owned context; never include a checkout or local Case artifacts.
    context = FRAMEWORK / 'target/image'
    if context.exists():
        shutil.rmtree(context)
    context.mkdir(parents=True)
    shutil.copyfile(jar, context / 'mock-engine.jar')
    for name in ('Dockerfile', 'whale_start.sh'):
        shutil.copyfile(FRAMEWORK / 'docker' / name, context / name)
    manifest = dict(schema_version=1, source_sha=source, recipe_sha=recipe,
                    jar_sha256=hashlib.sha256(jar.read_bytes()).hexdigest(), tests_executed=False,
                    internal_discovery=internal_discovery)
    (context / 'build-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)
    print(f'Docker context: {context}', flush=True)


if __name__ == '__main__':
    package()
