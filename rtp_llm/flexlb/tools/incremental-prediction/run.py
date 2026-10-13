#!/usr/bin/env python3
"""Usage: run.py BASELINE_FLEXLB_DIR OUTPUT_DIR; build both sync reactors before running."""
import json, os, pathlib, subprocess, sys, xml.etree.ElementTree as ET, statistics, re
root=pathlib.Path(__file__).resolve().parents[2]
baseline=pathlib.Path(sys.argv[1]).resolve(); out=pathlib.Path(sys.argv[2]).resolve(); out.mkdir(parents=True,exist_ok=True)
report=ET.parse(root/'flexlb-sync/target/surefire-reports/TEST-org.flexlb.balance.prediction.IncrementalPredictionTest.xml')
cp=report.find(".//property[@name='java.class.path']").attrib['value']
source=(root/'tools/incremental-prediction/PlanningBench.java').read_text()
formula=root/'flexlb-sync/src/test/resources/prediction/deepseek-v4.txt'
(out/'formula.txt').write_text(formula.read_text())
java=pathlib.Path(os.environ['JAVA_HOME'])/'bin'
for variant in ('before','after'):
    d=out/variant; d.mkdir(exist_ok=True)
    body=source
    if variant=='before':
        a=body.index('        // INCREMENTAL_BEGIN'); b=body.index('        // INCREMENTAL_END')+len('        // INCREMENTAL_END')
        body=body[:a]+'''        var selected = GroupPlanner.select(items, GroupPlanner.itemAccess(), constraints(size),
                prefix -> model.predictBatchMs(PrefillBatchFeatures.from(prefix,
                        GroupPlanner.Item::seqLen, GroupPlanner.Item::hitCache)));'''+body[b:]
    (d/'PlanningBench.java').write_text(body)
    runtime=cp.replace(str(root),str(baseline)) if variant=='before' else cp
    (d/'classpath.txt').write_text(runtime)
    subprocess.run([str(java/'javac'),'-proc:none','-cp',runtime,'-d',str(d),str(d/'PlanningBench.java')],check=True)
for fork in range(1,4):
    for variant in (('before','after') if fork%2 else ('after','before')):
        d=out/variant; runtime=(d/'classpath.txt').read_text()
        with (out/f'{variant}-{fork}.txt').open('w') as log:
            subprocess.run([str(java/'java'),'-Xms512m','-Xmx512m','-cp',str(d)+os.pathsep+runtime,
                            'PlanningBench',str(formula)],stdout=log,stderr=subprocess.STDOUT,check=True)
        print(f'{variant}-{fork} complete',flush=True)
data={}
for variant in ('before','after'):
    samples={}
    for p in out.glob(f'{variant}-*.txt'):
        for line in p.read_text().splitlines():
            m=re.match(r'(\w+) round=\d+ ns/op=([\d.]+) cpu_ns/op=([\d.]+) bytes/op=([\d.]+) checksum=([\d.]+)',line)
            if m:
                samples.setdefault(m[1],[]).append(list(map(float,m.groups()[1:])))
    data[variant]={k:dict(zip(('ns','cpu_ns','bytes','checksum'),map(statistics.median,zip(*v)))) for k,v in samples.items()}
(out/'summary.json').write_text(json.dumps(data,indent=2))
for k,b in data['before'].items():
    a=data['after'][k]
    if abs(a['checksum']-b['checksum'])>1e-6: raise AssertionError((k,a,b))
    print(k, 'speedup',round(b['ns']/a['ns'],2),'allocation_reduction',round(1-a['bytes']/b['bytes'],3))
