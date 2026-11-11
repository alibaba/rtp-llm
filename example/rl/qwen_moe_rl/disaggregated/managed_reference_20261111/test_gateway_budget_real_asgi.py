"""Exercise the pinned real gateway middleware's ASGI request-body path."""
import asyncio
import json
import os
import unittest
import rllm_model_gateway.middleware as actual
import q36_gateway_budget_compat as compat

compat.install(actual)

class FakeSessions:
    def __init__(self, data): self.data = data
    def get_sampling_params(self, sid): return self.data

async def invoke(body, session):
    forwarded = []
    async def app(scope, receive, send):
        forwarded.append(json.loads((await receive())['body']))
    async def receive(): return {'type':'http.request','body':json.dumps(body).encode(),'more_body':False}
    async def send(_): pass
    middleware=actual.SessionRoutingMiddleware(app,sessions=FakeSessions(session),model='fixed')
    await middleware({'type':'http','method':'POST','path':'/sessions/test/v1/chat/completions','state':{}},receive,send)
    return forwarded[0]

class RealGatewayBudget(unittest.TestCase):
    def test_client_and_session_minimum_in_real_asgi_path(self):
        for client,session,expected in [(1536,2048,1536),(512,2048,512),(4096,1536,1536),(None,1536,1536)]:
            body={'temperature':0.2}
            if client is not None: body['max_tokens']=client
            result=asyncio.run(invoke(body,{'max_tokens':session,'temperature':0.9,'top_p':0.8}))
            self.assertEqual(result['max_tokens'],expected)
            self.assertEqual(result['temperature'],0.9)
            self.assertEqual(result['top_p'],0.8)
            self.assertTrue(result['logprobs']);self.assertTrue(result['return_token_ids'])
            self.assertEqual(result['model'],'fixed')
    def test_absent_values_remain_absent_and_alias_works(self):
        self.assertNotIn('max_tokens',asyncio.run(invoke({},{})))
        a=asyncio.run(invoke({'max_completion_tokens':512},{'max_tokens':2048}))
        self.assertEqual(a['max_tokens'],512);self.assertNotIn('max_completion_tokens',a)
    def test_invalid_types_rejected(self):
        for val in (True,0,-1,1.5,'512'):
            with self.assertRaises(ValueError):asyncio.run(invoke({'max_tokens':val},{'max_tokens':2048}))
            with self.assertRaises(ValueError):asyncio.run(invoke({'max_tokens':512},{'max_tokens':val}))

if __name__=='__main__':unittest.main()
