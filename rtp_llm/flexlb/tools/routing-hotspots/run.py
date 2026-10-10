#!/usr/bin/env python3
"""Compare compiled revisions with identical snapshot/capture workloads.
Usage: JAVA_HOME=... python3 run.py BASELINE_FLEXLB_DIR OUTPUT_DIR
Build both revisions with test classes first. Metrics are per full-fleet scan;
64-planner wall time measures throughput, CPU time sums all measured threads.
"""
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[2]
baseline, output = (Path(value).resolve() for value in sys.argv[1:3])
output.mkdir(parents=True, exist_ok=True)
report = root / 'flexlb-sync/target/surefire-reports/TEST-org.flexlb.balance.prediction.IncrementalPredictionTest.xml'
classpath = ET.parse(report).find(".//property[@name='java.class.path']").attrib['value']
java = Path(os.environ['JAVA_HOME']) / 'bin'
source = Path(__file__).with_name('SnapshotBench.java')
verification = Path(__file__).with_name('ProjectionDifferential.java')
verification_signatures = []
formula = root / 'flexlb-sync/src/test/resources/prediction/deepseek-v4.txt'
for variant, directory in [('before', baseline), ('after', root)]:
    destination = output / variant
    destination.mkdir(exist_ok=True)
    runtime = classpath.replace(str(root), str(directory))
    (destination / 'classpath.txt').write_text(runtime)
    subprocess.run([str(java / 'javac'), '-proc:none', '-cp', runtime,
                    '-d', str(destination), str(source), str(verification)], check=True)
    verified = subprocess.check_output([str(java / 'java'), '-Xms512m', '-Xmx512m',
                                       '-cp', str(destination) + os.pathsep + runtime,
                                       'ProjectionDifferential', str(formula)], text=True)
    (destination / 'verification.txt').write_text(verified)
    verification_signatures.append([line for line in verified.splitlines() if line.startswith('VERIFY ')])
assert verification_signatures[0] and verification_signatures[0] == verification_signatures[1]
for fork in range(3):
    for variant in (['before', 'after'] if fork % 2 == 0 else ['after', 'before']):
        destination = output / variant
        runtime = str(destination) + os.pathsep + (destination / 'classpath.txt').read_text()
        with (output / f'{variant}-{fork}.txt').open('w') as log:
            subprocess.run([str(java / 'java'), '-Xms1g', '-Xmx1g', '-cp', runtime,
                            'SnapshotBench', str(formula)], stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        print(f'{variant}-{fork} complete', flush=True)
results = {}
signatures = []
for variant in ['before', 'after']:
    samples = {}
    for log in sorted(output.glob(f'{variant}-*.txt')):
        lines = log.read_text().splitlines()
        signatures.append([line for line in lines if line.startswith('VERIFY ')])
        for line in lines:
            match = re.fullmatch(r'(\w+) round=\d+ ns/op=([\d.]+) cpu_ns/op=([\d.]+) bytes/op=([\d.]+)', line)
            if match:
                samples.setdefault(match[1], []).append(tuple(map(float, match.groups()[1:])))
    assert len(samples) == 26 and all(len(values) == 9 for values in samples.values())
    results[variant] = {name: dict(zip(['ns', 'cpu_ns', 'bytes'], map(statistics.median, zip(*values))))
                        for name, values in samples.items()}
assert len(signatures) == 6 and signatures[0] and all(value == signatures[0] for value in signatures)
(output / 'summary.json').write_text(json.dumps(results, indent=2))
for name, before in results['before'].items():
    after = results['after'][name]
    print(name, 'speedup', round(before['ns'] / after['ns'], 2),
          'bytes_before', round(before['bytes'], 2), 'bytes_after', round(after['bytes'], 2))
