from pathlib import Path
import re,json
root=Path('/tmp/final-test-work'); removed={}
whole={'ObservationDisabledDoesNotStoreTokenSegments','ContentMatchIncludesTailAndPartialTokenInsideNode','ContentMatchInfoCarriesKeyOffsetAndPartialFlag','PrefixMatchObservationSeparatesContentAndReusableEndpoint','EvictionMutationBatchDoesNotRestartDependencyInterval','CompleteEvictObservesFinalTierStateAfterBatch','LoadMutationBatchDoesNotRestartDependencyInterval','RemoteObservationKeepsCandidateAndQueryStatusBeforeMaterialization','MatchQueryStatusIsDiagnosticAndDoesNotChangeSuccess'}
for p in (root/'rtp_llm').rglob('*'):
 if not p.is_file():continue
 s=p.read_text(); names=[]
 pattern=r'^TEST(?:_F)?\([^,]+,\s*(\w+)\) \{.*?^\}\n'
 def test(m):
  name=m[1]
  if name in whole or name.startswith(('PrefixObservability','PrefixObservation','PrefixPlan')):
   names.append(name);return ''
  return m[0]
 s=re.sub(pattern,test,s,flags=re.M|re.S)
 if names:removed[str(p.relative_to(root))]=names
 # Remove diagnostic helpers.
 s=re.sub(r'^    void (?:preparePrefixObservationResource|enablePrefixObservation|setNextQueryStatus)\(.*?^    \}\n','',s,flags=re.M|re.S)
 s=re.sub(r'^.*(?:enable_prefix_cache_observability\s*=|result\.query_status|next_query_status_\{|PrefixCacheRemoteQueryStatus\s+query_status\{|\.match_observation|resource.remote_observation_|TestLogCapture|PrefixCachePlanState|EXPECT_EQ\(plan_state,|contentMatchPos\().*\n','',s,flags=re.M)
 s=s.replace(', PrefixCacheRemoteQueryStatus)', ')')
 s=re.sub(r',\s*PrefixCacheRemoteQueryStatus query_status\)',')',s)
 s=s.replace('success, query_status}', 'success}')
 s=s.replace(', &plan_state)', ')')
 s=s.replace('false, nullptr, nullptr, nullptr, 771)', 'false)')
 s=re.sub(r',\s*/\*observability=\*/true','',s)
 s=re.sub(r'^.*bool\s+observability\s*= false,\n','',s,flags=re.M)
 s=re.sub(r'^\s*const std::string content = log_capture.content\(\);\n(?:.*<< content;\n)*','',s,flags=re.M)
 s=s.replace(',\n        std::vector<PrefixCachePlanPoint>{{0, 100, 1}})', ')')
 s=s.replace(', error, "copy_failed")', ', error)')
 s=re.sub(r'(BlockTree\s+tree\(.*), (?:true|false)\);',r'\1);',s)
 s=s.replace('std::move(group_sets), reuse_unit_tokens, config.enable_prefix_cache_observability)', 'std::move(group_sets), reuse_unit_tokens)')
 s=re.sub(r',\s*bool\s+enable_observability = false\)',')',s)
 s=s.replace('groups_, 1, enable_observability)', 'groups_, 1)')
 s=re.sub(r',\s*bool enable_prefix_observability = false\)',')',s)
 s=re.sub(r',\s*/\*enable_prefix_observability=\*/true','',s)
 s=s.replace('    for (const bool observation : {false, true}) {','    {')
 s=re.sub(r'        if \(observation\) \{\n            enablePrefixObservation\(\);\n        \}\n','',s)
 s=s.replace('cache->match({100, 200}, {}, &request_tokens)', 'cache->match({100, 200}, {})')
 s=s.replace('    const std::vector<int> request_tokens{1, 2, 3, 4};\n','')
 p.write_text(s)
(root/'removed.json').write_text(json.dumps(removed,indent=2))
