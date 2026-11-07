"""Offline proof-corruption tests; no Ray/Torch/optimizer is used."""
from pathlib import Path
import copy,hashlib,importlib.util,json,tempfile

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('validator',HERE/'validate_world72_core_evidence.py')
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)

def receipt(step,rows):
    padded=((rows+71)//72)*72
    return dict(schema='q38-real-single-global-minibatch-before-dispatch-v1',
        trainer_global_step=step,rows_pre_pad=rows,rows_post_pad=padded,global_minibatch_rows=padded,
        rollout_loss_denominator=72,data_parallel_size=72,rows_per_rank=padded//72,
        ppo_epochs=1,predicted_optimizer_steps_this_iteration=1,actual_distinct_role_losses=['vanilla'],
        original_backend_metadata_preserved=True,optimizer_step_executed_by_this_observer=False,
        pid=101,time_ns=1000+step)

def check(records,*,missing_update_rank=False):
    with tempfile.TemporaryDirectory(prefix='q38-offline-mini-proof-') as folder:
        root=Path(folder)
        for index,row in enumerate(records):
            (root/f'global-minibatch-step{row["trainer_global_step"]}-{index}.json').write_text(json.dumps(row))
        updates={s:{rank:(None,{}) for rank in range(72)} for s in (1,2)}
        if missing_update_rank:del updates[2][71]
        validator=MODULE.Validator([root]);validator.global_minibatches(updates)
        return validator.errors

baseline=[receipt(1,72),receipt(2,73)]
assert not check(baseline)
assert not check([receipt(1,143),receipt(2,144)])
cases=[]
for label,field,bad in [('wrong loss denominator','rollout_loss_denominator',144),
        ('two optimizer predictions','predicted_optimizer_steps_this_iteration',2),
        ('two PPO epochs','ppo_epochs',2),('wrong minibatch rows','global_minibatch_rows',72),
        ('non-neutral padding','rows_post_pad',216),('fake optimizer receipt','optimizer_step_executed_by_this_observer',True),
        ('multiple role losses','actual_distinct_role_losses',['a','b']),('too many turn rows','rows_pre_pad',145)]:
    candidate=copy.deepcopy(baseline);candidate[1][field]=bad
    assert check(candidate),label;cases.append(label)
assert check(baseline,missing_update_rank=True);cases.append('missing actual rank71 update commit')
assert check(baseline+[copy.deepcopy(baseline[0])]);cases.append('duplicate original dispatch')
assert check([receipt(0,72),receipt(2,72)]);cases.append('step0 cannot stand in for step1')
value=dict(schema='q38-offline-global-mini-proof-tests-v1',passed=True,
    positive_rows=[72,73,143,144],rejected_corruptions=cases,synthetic_fixture_only=True,
    actual_optimizer_or_RL_qualified=False,Ray_imported=False,Torch_imported=False,
    validator_sha256=hashlib.sha256((HERE/'validate_world72_core_evidence.py').read_bytes()).hexdigest())
out=HERE/'OFFLINE-MINI-PROOFS-CPU-QUALIFIED.json';out.write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value))
