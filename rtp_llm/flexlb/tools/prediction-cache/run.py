#!/usr/bin/env python3
"""Run identical source against baseline/current compiled reactors, alternating three JVM forks."""
import json, os, pathlib, re, statistics, subprocess, sys, xml.etree.ElementTree as ET
root = pathlib.Path(__file__).resolve().parents[2]
baseline, out = map(pathlib.Path, sys.argv[1:3])
out.mkdir(parents=True, exist_ok=True)
cp = ET.parse(root / 'flexlb-sync/target/surefire-reports/TEST-org.flexlb.balance.prediction.IncrementalPredictionTest.xml').find(".//property[@name='java.class.path']").attrib['value']
java = pathlib.Path(os.environ['JAVA_HOME']) / 'bin'
formula = root / 'flexlb-sync/src/test/resources/prediction/deepseek-v4.txt'
for variant, directory in [('before', baseline), ('after', root)]:
    dest = out / variant
    dest.mkdir(exist_ok=True)
    runtime = cp.replace(str(root), str(directory.resolve()))
    (dest / 'classpath.txt').write_text(runtime)
    sources = [pathlib.Path(__file__).with_name('CacheBench.java'),
               root/'tools/routing-hotspots/SnapshotBench.java',
               root/'tools/routing-hotspots/ProjectionDifferential.java']
    subprocess.run([str(java/'javac'), '-proc:none', '-cp', runtime, '-d', str(dest), *map(str, sources)], check=True)
    verified = subprocess.check_output([str(java/'java'), '-Xmx512m', '-cp', str(dest)+os.pathsep+runtime,
            'ProjectionDifferential', str(formula)], text=True)
    (dest/'differential.txt').write_text(verified)
assert [x for x in (out/'before/differential.txt').read_text().splitlines() if x.startswith('VERIFY ')] == [x for x in (out/'after/differential.txt').read_text().splitlines() if x.startswith('VERIFY ')]
for fork in range(3):
    for variant in (['before', 'after'] if fork % 2 == 0 else ['after', 'before']):
        dest = out / variant
        with (out / f'{variant}-{fork}.txt').open('w') as log:
            subprocess.run([str(java/'java'), '-Xms512m', '-Xmx512m', '-cp', str(dest)+os.pathsep+(dest/'classpath.txt').read_text(), 'CacheBench', str(formula)], stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'{variant}-{fork} complete', flush=True)
results = {}
for variant in ['before', 'after']:
    samples = {}
    for log in out.glob(f'{variant}-*.txt'):
        for line in log.read_text().splitlines():
            m = re.fullmatch(r'(\w+) ns=([\d.]+) cpu=([\d.]+) bytes=([\d.]+) checksum=([\d.]+)', line)
            if m: samples.setdefault(m[1], []).append(list(map(float, m.groups()[1:])))
    assert len(samples) == 6 and all(len(v) == 9 for v in samples.values())
    results[variant] = {k: dict(zip(['ns','cpu','bytes','checksum'], map(statistics.median, zip(*v)))) for k,v in samples.items()}
for name, before in results['before'].items():
    after = results['after'][name]
    assert before['checksum'] == after['checksum'], (name, before, after)
    print(name, 'cpu_reduction', round(1-after['cpu']/before['cpu'],3), 'wall_speedup', round(before['ns']/after['ns'],2))
(out/'summary.json').write_text(json.dumps(results, indent=2))
