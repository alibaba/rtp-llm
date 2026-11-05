from pathlib import Path
import re,subprocess,json,hashlib,difflib
root=Path('/tmp/final-test-work');repo='/data0/songchenyi.scy/Projects/RTP-LLM-20261024-Partial_Match/github-opensource';sha='1b56fd1e680b7842e3a1cfd246603d21a61102bf'
removed=json.loads((root/'removed.json').read_text()); report=[];diff=[];manifest=[];removed_assertions=[];total=0
for f in (root/'files').read_text().splitlines():
 old=subprocess.check_output(['git','-C',repo,'show',sha+':'+f],text=True);s=(root/f).read_text()
 assert not re.search(r'PrefixCachePlanState|PrefixCacheRemoteQueryStatus|prefix_cache_observability|match_observation|remote_observation_|contentMatchInfo|contentMatchPos|planRecoveryPoints|unfulfilledPlanPoints|prefixObservation|remoteObservation|capture\.content|^<<<<<<<|^=======|^>>>>>>>',s,re.M),f
 pat=r'^TEST(?:_F)?\([^,]+,\s*(\w+)\) \{.*?^\}\n';a={m[1]:m[0] for m in re.finditer(pat,old,re.M|re.S)};b={m[1]:m[0] for m in re.finditer(pat,s,re.M|re.S)}
 assert set(a)-set(b)==set(removed.get(f,[])),f
 assert not set(b)-set(a),f
 for n in a.keys()&b.keys():
  # Compare complete assertion statements, ignoring only whitespace changes.
  assertion=lambda x: [re.sub(r'\s+',' ',m[0]).strip() for m in re.finditer(r'(?:EXPECT|ASSERT)_\w+\(.*?;',x,re.S)]
  before=assertion(a[n]);after=assertion(b[n]);lost=[v for v in before if v not in after]
  for v in lost:
   assert any(k in v for k in ['match_observation','query_status','plan_state','logOccurrenceCount','content.find','contentMatchPos','remote_observation_','remoteObservation','summary.find','skipped.find','logs.find','first_publication']), (f,n,v)
  if lost:removed_assertions.append({'file':f,'test':n,'removed':lost})
 for n in ['TaskPhasePreventsDuplicateSettlementSubmission','RepeatedRunTransferCannotSettleSameTaskTwice']:
  if n in a:assert a[n]==b[n],n
 total+=len(b);report.append(f'{f}: PASS retained={len(b)} removed_whole={len(a)-len(b)}')
 diff.extend(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='a/'+f,tofile='b/'+f))
 manifest.append({'path':f,'sha256':hashlib.sha256(s.encode()).hexdigest(),'tests_retained':sorted(b),'whole_tests_removed':removed.get(f,[])})
(root/'final-no-metrics.patch').write_text(''.join(diff));(root/'manifest.json').write_text(json.dumps(manifest,indent=2));(root/'assertion-removals.json').write_text(json.dumps(removed_assertions,indent=2))
print('Invocation: python /tmp/final-test-work/verify.py')
print('Immutable basis:',sha)
print('\n'.join(report))
print('PASS: exact removed-whole-test inventory; no introduced tests; all removed assertions reference diagnostics; settlement tests byte-identical; diagnostic/conflict references absent.')
print('Retained tests:',total)
print('Build/runtime validation not run: preparation only; checked-out workspace not edited.')
