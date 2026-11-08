"""Exercise the exact candidate optimizer hook on a CPU-only bound stub.

This isolates source binding/order; it does not instantiate verl or CUDA and
must be combined with the separately qualified real 4-GPU optimizer fixture.
"""
import ast,math,os,sys,types,unittest
from pathlib import Path
from unittest.mock import patch

SOURCE=Path(__file__).with_name('qwen38_bf16_training_runtime.py')

class HookBindingTest(unittest.TestCase):
    def setUp(self):
        raw=SOURCE.read_bytes()
        tree=ast.parse(raw);install=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='install')
        step=next(x for x in install.body if isinstance(x,ast.FunctionDef) and x.name=='step')
        helper=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_optimizer_step_after_global_clip')
        assignment=[x for x in install.body if isinstance(x,ast.Assign) and any(
            isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='Engine'
            and t.attr=='optimizer_step' for t in x.targets)]
        self.assertEqual(len(assignment),1)
        self.assertIsInstance(assignment[0].value,ast.Name);self.assertEqual(assignment[0].value.id,'step')
        self.events=[];events=self.events
        class Tensor:
            def __init__(self,value):self.value=value
            def item(self):return self.value
        class Dist:
            ReduceOp=types.SimpleNamespace(SUM='sum')
            @staticmethod
            def is_initialized():return True
            @staticmethod
            def get_world_size():return 72
            @staticmethod
            def all_reduce(_tensor,op):events.append('reduce-changed')
        torch=types.SimpleNamespace(device=lambda *args:args,cuda=types.SimpleNamespace(current_device=lambda:0),
            int64='int64',tensor=lambda value,**kwargs:Tensor(value))
        namespace=dict(os=os,math=math,torch=torch,dist=Dist,WORLD_SIZE=72,POLICY_NUMEL=123,
            _is_target=lambda engine:engine.target,original_step=lambda engine:'original',
            _audit=lambda _engine,boundary:events.append('audit-'+boundary))
        module=ast.fix_missing_locations(ast.Module(body=[helper,step],type_ignores=[]))
        exec(compile(module,str(SOURCE),'exec'),namespace)
        class Engine:
            optimizer_step=namespace['step']
        self.engine=Engine();self.engine.target=True;self.engine.scaler=None
        self.engine._q38_first_adam_admitted=False;self.engine._q38_owned_numel=9
        self.engine._q38_cpu_admission_group='group'
        self.engine.optimizer_config=types.SimpleNamespace(clip_grad=1)
        self.engine._q38_phases=types.SimpleNamespace(require_training=lambda:events.append('training-phase'))
        self.engine._q38_gradient_store=types.SimpleNamespace(clip_grad_norm=lambda *a,**k:(events.append('clip') or 1.5))
        self.engine.optimizer=types.SimpleNamespace(changed_weight_elements=3,
            step=lambda:events.append('optimizer-step'),zero_grad=lambda:events.append('zero-grad'))

    def gate_module(self,deny=False):
        def first(engine,dist,group,world):
            self.events.append('admission')
            self.assertIs(engine,self.engine);self.assertEqual(group,'group');self.assertEqual(world,72)
            if deny:raise MemoryError('fixture denied')
        return types.SimpleNamespace(first_adam_admission=first)

    def test_denied_gate_precedes_and_prevents_optimizer_update(self):
        with patch.dict(os.environ,{'Q38_CPU_FIRST_ADAM_ADMISSION_REQUIRED':'1','RLLM_REQUIRE_REAL_UPDATE':'1'}), \
             patch.dict(sys.modules,{'distributed_admission':self.gate_module(deny=True)}):
            with self.assertRaises(MemoryError):self.engine.optimizer_step()
        self.assertEqual(self.events,['training-phase','clip','admission'])
        self.assertFalse(self.engine._q38_first_adam_admitted)

    def test_passed_gate_updates_and_next_step_does_not_reenter_first_gate(self):
        with patch.dict(os.environ,{'Q38_CPU_FIRST_ADAM_ADMISSION_REQUIRED':'1','RLLM_REQUIRE_REAL_UPDATE':'1'}), \
             patch.dict(sys.modules,{'distributed_admission':self.gate_module()}):
            self.assertEqual(self.engine.optimizer_step(),1.5)
            self.assertTrue(self.engine._q38_first_adam_admitted)
            self.engine.optimizer_step()
        self.assertEqual(self.events,['training-phase','clip','admission','optimizer-step','reduce-changed',
            'audit-optimizer-step','zero-grad','training-phase','clip','optimizer-step','reduce-changed',
            'audit-optimizer-step','zero-grad'])

    def test_scaler_and_non_target_paths(self):
        self.engine.scaler=object()
        with self.assertRaises(RuntimeError):self.engine.optimizer_step()
        self.assertEqual(self.events,['training-phase'])
        self.events.clear();self.engine.target=False
        self.assertEqual(self.engine.optimizer_step(),'original');self.assertEqual(self.events,[])

if __name__=='__main__':unittest.main()
