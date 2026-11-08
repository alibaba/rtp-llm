"""Exercise the actual 72-world launcher with stubbed Ray actor construction."""
import ast
import os
from pathlib import Path
import sys
import types

source=(Path(__file__).resolve().parents[1]/'adapter/qwen38_world_server.py').read_text()
tree=ast.parse(source)
method=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='launch_world72_server')
module=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
seen=[]
class Actor:
    def options(self,**kwargs):seen.append(kwargs);return self
    def remote(self,*args):return object()
class Ray:
    def is_initialized(self):return True
    def remote(self,**kwargs):return lambda cls:Actor()
class Topology:
    @classmethod
    def from_dict(cls,value):return cls()
    def to_dict(self):return {}
strategy=types.ModuleType('ray.util.scheduling_strategies')
strategy.NodeAffinitySchedulingStrategy=lambda node_id,soft: (node_id,soft)
sys.modules['ray']=types.ModuleType('ray')
sys.modules['ray.util']=types.ModuleType('ray.util')
sys.modules['ray.util.scheduling_strategies']=strategy
scope=dict(os=os,ray=Ray(),World72Topology=Topology,WorldNodeOwner=object,World72RTPServer=object,
           require=lambda condition,message: None if condition else (_ for _ in ()).throw(AssertionError(message)))
exec(compile(module,'actual-launch-world72-AST','exec'),scope)
for value,expected in [(None,'0'),('1','1')]:
    if value is None:os.environ.pop('Q38_LOGPROB_REPORT_ONLY',None)
    else:os.environ['Q38_LOGPROB_REPORT_ONLY']=value
    seen.clear();scope['launch_world72_server']('',{},[str(i) for i in range(18)])
    assert len(seen)==19
    assert all(row['runtime_env']['env_vars']['Q38_LOGPROB_REPORT_ONLY']==expected for row in seen)
print('WORLD72_LOGPROB_ENV_CPU_PASS: both explicit task mode and default strict reach all 19 actor options')
