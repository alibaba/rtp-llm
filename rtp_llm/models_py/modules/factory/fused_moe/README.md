# Fused-MoE factory

The factory combines a strategy, router and executor through a model-independent
configuration adapter. CUDA FP8/FP4 backends are not specific to DeepSeek-V4.

## Layout

```text
fused_moe/
├── __init__.py           # Public exports and device-specific registration
├── factory.py            # FusedMoeFactory
├── strategy_registry.py  # Candidate filtering and selection
├── defs/
│   ├── config_adapter.py # MoEConfigAdapter
│   ├── fused_moe.py      # FusedMoe, router/executor and payload contracts
│   ├── strategy_base.py  # MoeStrategy
│   ├── priority_attributes.py
│   ├── quant_config.py
│   └── type.py
├── impl/
│   ├── common/strategy/  # Batched Triton fallback
│   ├── cuda/
│   │   ├── strategy/     # Includes fp8_fp4.py
│   │   ├── routers/
│   │   └── executors/
│   └── rocm/
│       ├── strategy/
│       ├── routers/
│       └── executors/
├── utils/
│   ├── condition_checker.py
│   ├── config_resolver.py
│   ├── fp8_fp4/          # Layer, gate and chunking orchestration
│   ├── mega_moe/         # Buffers, packing and warmup helpers
│   └── test/
└── tests/                # Factory/registry tests
```

Component tests also live in `defs/test/` and the relevant `impl/*/.../test/`
directories. GPU kernels live outside this package, under
`rtp_llm/models_py/kernels/cuda/` and `rtp_llm/models_py/triton_kernels/moe/`.
DeepEP initialization remains in
`rtp_llm/models_py/distributed/deepep_initializer.py`.

## Creation and selection

```python
from rtp_llm.models_py.modules.factory.fused_moe import FusedMoeFactory

# config is a MoEConfigAdapter; weights is the per-layer framework weight dict.
moe = FusedMoeFactory().create_fused_moe(config, weights)
```

Importing this package registers in-tree strategies in `__init__.py`, selects
the device branch, and installs its registry with
`FusedMoeFactory.set_registry()`. There is no separate CUDA/ROCm registry module.
Out-of-tree registration hooks run afterwards through
`run_backend_registrations("fused_moe", registry=registry)`.

The registry filters candidates by quantization contract and the requested
`moe_strategy`, then calls `can_handle()` and chooses the highest priority.
Equal-priority candidates retain registration order. The priority is derived
from the router/executor types in `StrategyAttributes`.

For eligible CUDA FP8/FP4 configurations, `auto` prefers `mega_moe_se` with
shared experts, otherwise `mega_moe`. Single-rank alternatives are
`grouped_fp4` and `local_loop`. Actual availability also depends on device,
kernel, quantization, parallelism and shared-expert gate support.
`mega_moe_se` supports multiple shared experts; it does not require exactly one.
Explicit selection uses `--moe_strategy` or `MOE_STRATEGY`.

`FusedMoe` coordinates router dispatch, expert execution and output combine.
The optional `ExpertGatePayload` / `supports_gate_pack` contract allows a
capable router/executor pair to fuse routing with input packing; unsupported
calls continue through materialized top-k ids and weights.

## Extending the factory

1. Put a router in `impl/<device>/routers/` and an executor in
   `impl/<device>/executors/`. Implement the contracts in
   `defs/fused_moe.py`, including their type and capability checks.
2. Add a strategy under `impl/<device>/strategy/`, deriving from
   `rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base.MoeStrategy`.
   Implement `get_attributes()` with the router class, executor class and
   `FusedMoEQuantConfig`. Override creation methods only when the default
   `(config, quant_config[, weights])` constructors are insufficient.
3. Declare `strategy_name` for exact-name selection and
   `supported_moe_quant_method = "FP8_FP4"` for an FP8/FP4 strategy.
   Use `check_conditions()` for strategy-specific constraints; the base
   `can_handle()` also checks the router and executor.
4. Export the strategy in `impl/<device>/strategy/__init__.py` and register
   its instance in this package's `__init__.py`, within the correct device
   branch. Keep CUDA/DeepGEMM imports deferred to the appropriate backend path.
   See `impl/cuda/strategy/fp8_fp4.py` for existing examples.
5. Add focused tests to the corresponding component's `test/BUILD` and
   selection tests to `tests/BUILD`. Cover capability rejection, fallback,
   shared/routed-only behavior and actual execution for affected GPU paths.

## Migration

The old `rtp_llm.models_py.modules.dsv4.moe` package has been removed without a
compatibility shim. Callers must migrate to the generic factory or these
explicit interfaces under `rtp_llm.models_py.modules.factory.fused_moe`:

- `utils.fp8_fp4.layer.Fp8Fp4MoeLayer`: explicit layer parameters and weights.
- `utils.fp8_fp4.gate.Gate`: routing gate.
- `impl.cuda.executors.grouped_fp4._has_fp8_fp4_grouped_kernel`: kernel probe.

## Tests

Factory tests are declared in `tests/BUILD`, including
`test_strategy_registry` and `test_strategy_select`.
Use the repository's test-execution workflow with the matching platform and
existing cache configuration. CUDA13 SM100 tests have separate ARM and x86
targets; select the target matching the build and execution platform.
