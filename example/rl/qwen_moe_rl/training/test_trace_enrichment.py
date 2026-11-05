"""CPU regressions for delivered-response association, with no rLLM/CUDA import."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

SPEC = importlib.util.spec_from_file_location('qwen36_trace_enrichment', Path(__file__).parent/'overrides/qwen36_trace_enrichment.py')
overlay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(overlay)


def trace(identity, request, assistant, uid='session:0'):
    return NS(trace_id=identity, session_id=uid, messages=copy.deepcopy(request),
        response_message=copy.deepcopy(assistant), prompt_token_ids=[1,2,3],
        completion_token_ids=[4,5], logprobs=[-0.25,-0.5], raw_response={'id':'response-'+identity},
        weight_version=1, timestamp=123.0)


def episode(histories):
    return NS(trajectories=[NS(steps=[NS(chat_completions=copy.deepcopy(history),
        model_response=history[-1].get('content') or '', action=history[-1].get('content') or '') for history in histories])],
        artifacts={'answer':'42','final_content':histories[-1][-1].get('content') or ''})


class TraceLinkageTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.previous=os.environ.get('RLLM_QWEN36_TRACE_AUDIT_DIR')
        os.environ['RLLM_QWEN36_TRACE_AUDIT_DIR']=self.directory.name
        self.received=[]
        class Mismatch(RuntimeError):pass
        def original(ep,traces,uid,task,*,strict=True):
            self.received.append((ep,list(traces),uid,task,strict))
            return {'episode':ep,'traces':traces}
        self.module=NS(enrich_episode_with_traces=original,EnrichMismatchError=Mismatch)
        overlay.install(self.module)
        self.request=[{'role':'system','content':'Solve.'},{'role':'user','content':'6 * 7?'}]

    def tearDown(self):
        if self.previous is None:os.environ.pop('RLLM_QWEN36_TRACE_AUDIT_DIR',None)
        else:os.environ['RLLM_QWEN36_TRACE_AUDIT_DIR']=self.previous
        self.directory.cleanup()

    def call(self,ep,traces):
        return self.module.enrich_episode_with_traces(ep,traces,'session:0',{'question':'6 * 7?'})

    def audit(self):
        files=list(Path(self.directory.name).glob('*.json'))
        self.assertEqual(len(files),1)
        return json.loads(files[0].read_text())

    def test_retry_selects_delivered_response_and_preserves_objects(self):
        delivered={'role':'assistant','content':'The answer is \\boxed{42}.'}
        orphan=trace('timed-out',self.request,{'role':'assistant','content':'Different unfinished response.'})
        selected=trace('delivered',self.request,delivered)
        ep=episode([self.request+[delivered]])
        before=copy.deepcopy(ep.artifacts)
        output=self.call(ep,[orphan,selected])
        self.assertIs(output['traces'][0],selected)
        self.assertIs(output['traces'][0].logprobs,selected.logprobs)
        self.assertEqual(ep.artifacts,before)
        self.assertEqual(ep.trajectories[0].steps[0].action,delivered['content'])
        audit=self.audit()
        self.assertTrue(audit['passed']);self.assertEqual(audit['selected'][0]['trace_id'],'delivered')
        self.assertEqual(audit['orphaned'][0]['trace_id'],'timed-out')
        self.assertEqual(len(audit['selected'][0]['raw_logprobs_fp32_sha256']),64)

    def test_out_of_order_multi_turn_traces(self):
        first={'role':'assistant','content':'First response.'}
        second_request=self.request+[first,{'role':'user','content':'Explain.'}]
        second={'role':'assistant','content':'Second response.'}
        t1=trace('turn1',self.request,first);t2=trace('turn2',second_request,second)
        output=self.call(episode([self.request+[first],second_request+[second]]),[t2,t1])
        self.assertEqual([x.trace_id for x in output['traces']],['turn1','turn2'])

    def test_tool_id_and_argument_identity(self):
        assistant={'role':'assistant','content':'','tool_calls':[{'id':'call-good','type':'function',
            'function':{'name':'python','arguments':'{"code":"print(6 * 7)"}'}}]}
        wrong=copy.deepcopy(assistant);wrong['tool_calls'][0]['id']='call-wrong'
        t1=trace('wrong-tool-id',self.request,wrong);t2=trace('correct-tool-id',self.request,assistant)
        # SDK and trace may represent the same JSON arguments as string/dict.
        t2.response_message['tool_calls'][0]['function']['arguments']={'code':'print(6 * 7)'}
        output=self.call(episode([self.request+[assistant]]),[t1,t2])
        self.assertIs(output['traces'][0],t2)
        self.assertEqual(self.audit()['selected'][0]['tool_call_ids'],['call-good'])

    def test_missing_delivered_response_fails(self):
        ep=episode([self.request+[{'role':'assistant','content':'Actual response.'}]])
        with self.assertRaises(self.module.EnrichMismatchError):
            self.call(ep,[trace('different',self.request,{'role':'assistant','content':'Wrong response.'})])
        self.assertFalse(self.audit()['passed']);self.assertEqual(self.received,[])

    def test_identical_candidates_are_ambiguous(self):
        response={'role':'assistant','content':'42'}
        with self.assertRaises(self.module.EnrichMismatchError):
            self.call(episode([self.request+[response]]),[trace('a',self.request,response),trace('b',self.request,response)])
        self.assertFalse(self.audit()['passed']);self.assertEqual(self.received,[])

    def test_wrong_tool_code_is_not_a_match(self):
        response={'role':'assistant','content':'','tool_calls':[{'id':'call-1','type':'function',
            'function':{'name':'python','arguments':'{"code":"print(42)"}'}}]}
        wrong=copy.deepcopy(response);wrong['tool_calls'][0]['function']['arguments']='{"code":"print(24)"}'
        with self.assertRaises(self.module.EnrichMismatchError):
            self.call(episode([self.request+[response]]),[trace('wrong-code',self.request,wrong)])
        self.assertFalse(self.audit()['passed'])

    def test_wrong_prior_tool_return_history_is_not_a_match(self):
        request=self.request+[{'role':'assistant','content':'','tool_calls':[{'id':'call-1',
            'function':{'name':'python','arguments':'{"code":"print(42)"}'}}]},
            {'role':'tool','tool_call_id':'call-1','content':'{"stdout":"42"}'}]
        response={'role':'assistant','content':'42'}
        wrong=copy.deepcopy(request);wrong[-1]['content']='{"stdout":"24"}'
        with self.assertRaises(self.module.EnrichMismatchError):
            self.call(episode([request+[response]]),[trace('wrong-history',wrong,response)])
        self.assertFalse(self.audit()['passed'])

    def test_unrelated_extra_call_cannot_be_discarded(self):
        response={'role':'assistant','content':'42'}
        extra_request=self.request+[response,{'role':'user','content':'Unrecorded turn.'}]
        with self.assertRaises(self.module.EnrichMismatchError):
            self.call(episode([self.request+[response]]),[trace('good',self.request,response),
                trace('unrecorded',extra_request,{'role':'assistant','content':'Unrecorded response.'})])
        self.assertFalse(self.audit()['passed']);self.assertEqual(self.received,[])

    def test_cross_session_trace_fails_even_with_matching_text(self):
        response={'role':'assistant','content':'42'}
        with self.assertRaises(self.module.EnrichMismatchError):
            self.call(episode([self.request+[response]]),[trace('foreign',self.request,response,uid='session:1')])
        self.assertFalse(self.audit()['passed'])

    def test_install_is_idempotent(self):
        first=self.module.enrich_episode_with_traces
        overlay.install(self.module)
        self.assertIs(first,self.module.enrich_episode_with_traces)


if __name__=='__main__':unittest.main(verbosity=2)
