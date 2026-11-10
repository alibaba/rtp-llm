import types
import unittest
from q36_nonshared_bf16_worker_candidate import install_hooks


class Flag:
    def __init__(self, value): self.value = value
    def item(self): return self.value


class Parameter:
    requires_grad = True


def make_modules():
    def init(optim_config, use_distributed_optimizer=True, fp16=False, bf16=True):
        return types.SimpleNamespace(optimizer='adam', bf16=True, use_distributed_optimizer=False,
            use_precision_aware_optimizer=False, optimizer_cpu_offload=True, optimizer_offload_fraction=1.0,
            overlap_cpu_optimizer_d2h_h2d=False, overlap_param_gather_with_optimizer_step=False)
    verl = types.SimpleNamespace(init_megatron_optim_config=init, get_megatron_optimizer_native=None)
    core = types.SimpleNamespace(_q38_candidate_installed=False)
    def native(config, model_chunks, *args, **kwargs): return config.q38_sr_bindings
    def install(): core._q38_candidate_installed = True; core.get_megatron_optimizer = native
    factory = types.SimpleNamespace(install=install)
    torch = types.SimpleNamespace(bfloat16='bf16', int32='i32', cuda=types.SimpleNamespace(current_device=lambda: 0),
                                  tensor=lambda value, **_: Flag(value))
    votes = []
    dist = types.SimpleNamespace(ReduceOp=types.SimpleNamespace(MIN='min'),
                                 all_reduce=lambda flag, op: votes.append(flag.item()))
    return verl, core, factory, torch, dist, votes


class InjectionTests(unittest.TestCase):
    def test_actual_chunk_ids_are_injected_without_shared_lease(self):
        verl, core, factory, torch, dist, votes = make_modules()
        param = Parameter(); chunk = types.SimpleNamespace(parameters=lambda: [param])
        install_hooks(verl, core, factory, torch, dist, lambda chunks: {id(param): 'coordinate'})
        config = verl.init_megatron_optim_config({}, use_distributed_optimizer=False)
        self.assertEqual(config.main_grads_dtype, 'bf16')
        self.assertFalse(hasattr(config, 'q38_shared_lease'))
        self.assertEqual(verl.get_megatron_optimizer_native(config, [chunk]), {id(param): 'coordinate'})
        self.assertEqual(votes, [1])

    def test_missing_binding_votes_failure_before_native_factory(self):
        verl, core, factory, torch, dist, votes = make_modules()
        param = Parameter(); chunk = types.SimpleNamespace(parameters=lambda: [param])
        install_hooks(verl, core, factory, torch, dist, lambda chunks: {})
        config = verl.init_megatron_optim_config({}, use_distributed_optimizer=False)
        with self.assertRaisesRegex(RuntimeError, 'all-rank'):
            verl.get_megatron_optimizer_native(config, [chunk])
        self.assertEqual(votes, [0])

    def test_missing_mapper_is_rejected(self):
        verl, core, factory, torch, dist, _ = make_modules()
        with self.assertRaisesRegex(TypeError, 'mapper'):
            install_hooks(verl, core, factory, torch, dist, None)


if __name__ == '__main__': unittest.main()
