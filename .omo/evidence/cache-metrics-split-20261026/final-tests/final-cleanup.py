from pathlib import Path
import re
root=Path('/tmp/final-test-work')
for f in (root/'files').read_text().splitlines():
 p=root/f;s=p.read_text()
 s=re.sub(r'^size_t logOccurrenceCount\(.*?^}\n','',s,flags=re.M|re.S)
 s=s.replace('    for (const bool observe : {false, true}) {','    {')
 s=s.replace('        if (observe) {\n        } else {\n        }\n','')
 s=re.sub(r'^.*(?:enablePrefixObservation\(\)|prefixObservationSummaryPayload|remoteObservation\().*\n','',s,flags=re.M)
 s=re.sub(r'^\s*(?:EXPECT|ASSERT)_\w+\((?:summary\.find|skipped\.find|logs\.find|first_publication).*?;\n','',s,flags=re.M|re.S)
 s=re.sub(r'^.*const auto (?:skipped|logs|first_publication) = .*\n','',s,flags=re.M)
 s=s.replace(', PrefixCacheRemoteQueryStatus::HIT}', '}')
 # Remove only obsolete observation argument between validity predicate and partial-tail request.
 s=re.sub(r'(->match\([^;\n]*?, \{\}), (?:nullptr|&\w+), (&\w+\))',r'\1, \2',s)
 # Remove diagnostic plan-state argument between token segments and endpoints.
 s=re.sub(r'(&\w+),\s*nullptr,(\s*&\w+)',r'\1,\2',s)
 # Remove request observation id preceding functional internal-fork arguments.
 s=re.sub(r',\s*std::nullopt,(\s*(?:/\*internal_fork=\*/)?(?:true|false))',r',\1',s)
 s=s.replace('std::chrono::seconds(30), std::vector<std::vector<int>>{{1}})', 'std::chrono::seconds(30))')
 p.write_text(s)
