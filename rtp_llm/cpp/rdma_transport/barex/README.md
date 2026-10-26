# RDMA transport implementation (barex/accl)

This directory holds the RDMA implementation of the public Tensor RDMA
transport interface (`rtp_llm/cpp/rdma_transport/RdmaTransport.h`), together
with the open-sourced RDMA variants of the cache store
(`rtp_llm/cpp/disaggregate/cache_store/rdma/`) and the P2P transfer backend
(`rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/`). The RDMA support
layer they build on (`aios/network/rdma/`: the accl mempool wrapper, the
arpc-over-RDMA channel, and the python bindings) is open-sourced in the same
tree at `aios/network/rdma/`.

## Gating

All of these implementation targets are selected through
`arch_config/arch_select.bzl` (`cache_store_deps` / `rdma_transport_deps` /
`transfer_backend_deps`) and keyed on the `use_rdma` define:

```bash
bazel build --define=use_rdma=true //rtp_llm/cpp/rdma_transport/barex:barex_rdma_transport_impl
```

`use_rdma` is **OFF by default and no public platform configuration enables
it**. Every target that links the closed accl-barex library — directly or
through `//aios/network/rdma` — gates its `srcs`/`hdrs`/`deps` on
`//:using_rdma`, so a default public `bazel build //...` compiles empty
libraries and never fetches `@alibaba_rdma`.

## accl stays closed

accl-barex (ACCBL) is a **permanently closed** library. The
`@alibaba_rdma` repository rule and the accl/solar/tnet/ali-rdma-core RPMs
are defined only in the internal deps overlay — the public tree must never
gain a fetch rule for them. `aios/network/rdma` keeps only one pointer into
that closed world: the accl-barex labels inside the `if_rdma` ON branches
(`@alibaba_rdma//aios/network/accl-barex/...`).

Building with `--define=use_rdma=true` **requires the internal overlay**.
Without it the build fails with an error like:

```
ERROR: no such package '@alibaba_rdma//aios/network/accl-barex': ...
```

(Reached through this target's direct
`@alibaba_rdma//aios/network/accl-barex:accl-barex` dep; before
`aios/network/rdma` itself was open-sourced the same failure surfaced one
hop earlier, on the `@alibaba_rdma//aios/network/rdma` package.)

This failure is intentional: it is the same mechanism that keeps
`//3rdparty/tnet` dangling-but-harmless in the public tree.

## Tests

The RDMA tests moved alongside the implementation are tagged `manual` (plus
their GPU pool tag, e.g. `L20_CU13`/`H20_CU13`): public wildcard lanes never
schedule them. They need real RDMA NICs (e.g. `ACCL_USE_NICS=mlx5_bond_0`)
and are run explicitly on internal RDMA workers. The `aios/network/rdma`
unit/smoke tests follow the same rule.
