"""CPU regression for shared HF consumers, ownership and bounded buffering."""
import ast
import json
from pathlib import Path
import types
from typing import *

import torch

HERE=Path(__file__).resolve().parent


def real_definitions(path, name, selected=None):
    tree=ast.parse(path.read_text())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name==name)
    if selected is not None:
        cls=ast.ClassDef(name=name,bases=[],keywords=[],body=[node for node in cls.body
                         if isinstance(node,ast.FunctionDef) and node.name in selected],decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[]))
    env=dict(globals(),TensorSource=object,BaseDatabase=object)
    exec(compile(module,str(path),'exec'),env)
    return env[name]


def main():
    from qwen36_overlay import install
    install()
    from rtp_llm.model_loader.ffn_weight import iter_stacked_moe_weights
    globals()['iter_stacked_moe_weights']=iter_stacked_moe_weights
    collector=real_definitions(HERE/'original_tensor_source.py','TensorCollector')
    globals()['TensorCollector']=collector
    loader_cls=real_definitions(HERE/'overlays/loader.py','ModelLoader',
                               {'prepare_weights_from_tensors','get_rl_hf_source_names',
                                '_get_rl_group_source_names'})
    class Descriptor:
        def __init__(self,name,names,fn):self.name=name;self.names=set(names);self.fn=fn
        def get_tensor_names(self,layer,config):return self.names
        def load(self,source,layer,device,config):return {self.name:self.fn(source._tensors)}
    descriptors=[Descriptor('effective_gamma',{'gamma'},lambda ts:ts['gamma']+1),
                 Descriptor('raw_gamma_and_gate',{'gamma','gate'},lambda ts:ts['gamma']+ts['gate']),
                 Descriptor('synthesized_bias',set(),lambda ts:torch.zeros(4,dtype=torch.bfloat16))]
    Group=NamedTuple('Group',[('weight',object),('layer_id',Optional[int]),('collector',object)])
    def make():
        obj=loader_cls();obj._load_config=types.SimpleNamespace(is_ft_style_weight=False,merge_lora=False,database=None)
        def groups():
            items=[Group(d,0,collector(d.names,None)) for d in descriptors]
            return {},items
        obj._generate_weight_info=groups;return obj
    gamma=torch.tensor([0,1,2,3],dtype=torch.bfloat16);gate=torch.full((4,),2,dtype=torch.bfloat16)
    result={name:value for layer,name,value in make().prepare_weights_from_tensors(
        [('gamma',gamma),('gate',gate)],'cpu',max_buffer_bytes=16)}
    assert torch.equal(result['effective_gamma'],torch.tensor([1,2,3,4],dtype=torch.bfloat16))
    assert torch.equal(result['raw_gamma_and_gate'],torch.tensor([2,3,4,5],dtype=torch.bfloat16))
    assert torch.equal(gamma,torch.tensor([0,1,2,3],dtype=torch.bfloat16))
    assert torch.count_nonzero(result['synthesized_bias']).item()==0
    assert make().get_rl_hf_source_names()==('gamma','gate')
    for inputs,limit,text in [([('gamma',gamma),('gate',gate)],15,'buffer exceeds'),
                              ([('gamma',gamma)],16,'missing HF tensors'),
                              ([('gamma',gamma),('gamma',gamma)],16,'duplicate HF tensor'),
                              ([('unexpected',gamma)],16,'unexpected HF tensor')]:
        try:list(make().prepare_weights_from_tensors(inputs,'cpu',max_buffer_bytes=limit))
        except (ValueError,KeyError) as e:assert text in str(e),(text,e)
        else:raise AssertionError('expected '+text)
    print(json.dumps({'passed':True,'cases':5,'device':'cpu','shared_input_clone_counted_once':True}))


if __name__=='__main__':main()
