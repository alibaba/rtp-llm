import pathlib,subprocess,re,json
r=pathlib.Path('/data0/songchenyi.scy/Projects/RTP-LLM-20261024-Partial_Match/github-opensource');here=pathlib.Path(__file__).parent;files=(here/'files').read_text().splitlines(); removed=json.loads((here/'removed.json').read_text());count=0
for f in files:
 s=(r/f).read_text();old=subprocess.check_output(['git','-C',str(r),'show','61acca1d5e3a91c01df22be6a6fdae7987dc2550:'+f],text=True)
 assert not re.search(r'PrefixCachePlanState|PrefixCacheRemoteQueryStatus|prefix_cache_observability|match_observation|remote_observation_|contentMatchInfo|contentMatchPos|planRecoveryPoints|unfulfilledPlanPoints|^<<<<<<<|^=======|^>>>>>>>',s,re.M),f
 pat=r'^TEST(?:_F)?\([^,]+,\s*(\w+)\) \{.*?^\}\n';a={m[1]:m[0] for m in re.finditer(pat,old,re.M|re.S)};b={m[1]:m[0] for m in re.finditer(pat,s,re.M|re.S)}
 assert set(a)-set(b)==set(removed.get(f,[])),f
 assert not set(b)-set(a),f
 for name in ['TaskPhasePreventsDuplicateSettlementSubmission','RepeatedRunTransferCannotSettleSameTaskTwice']:
  if name in a: assert a[name]==b[name],name
 count+=len(b)
 print(f, 'PASS; tests retained:',len(b),'whole diagnostics removed:',len(a)-len(b))
res=subprocess.run(['git','-C',str(r),'diff','--check','--',*files],capture_output=True,text=True);assert res.returncode==0,res.stdout+res.stderr
print('PASS: diagnostic/conflict scan; exact whole-test inventory; unchanged settlement tests; diff --check. Retained tests:',count)
print('Build/runtime tests NOT RUN: delegated owner integrates source and runs builds.')
