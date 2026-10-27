"""RDMA arch-selects, shared verbatim by the public tree and the internal overlay.

The internal overlay overrides the @arch_config repository with
internal_source/bazel, which shadows the whole public arch_config/ directory
(Bazel refuses main-repo labels into it: it is a repository root, not a
package). The internal arch_select.bzl therefore cannot load
arch_config/arch_select.bzl — these definitions live in the main repo's
bazel/ package instead, which is loadable from any repository, and both
arch_select.bzl files re-export them.
"""

def cache_store_deps():
    # using_rdma is default OFF and no public platform config enables it: the
    # open-source build selects the TCP base implementation. The RDMA variant
    # (cache_store_rdma_impl, links the closed accl-barex library via
    # @alibaba_rdma) is selected only by internal overlay builds, which set
    # --define=use_rdma=true.
    native.alias(
        name = "cache_store_arch_select_impl",
        actual = select({
            "@rtp_llm//:using_rdma": "@rtp_llm//rtp_llm/cpp/disaggregate/cache_store/rdma:cache_store_rdma_impl",
            "//conditions:default": "@rtp_llm//rtp_llm/cpp/disaggregate/cache_store:cache_store_base_impl",
        }),
    )

def rdma_transport_deps():
    # using_rdma is default OFF and no public platform config enables it: the
    # open-source build links the no-impl stub (the factory reports RDMA as
    # unavailable). The barex provider (links the closed accl-barex library
    # via @alibaba_rdma) is selected only by internal overlay builds, which
    # set --define=use_rdma=true.
    #
    # Within the ON configuration the platform matrix is the pre-migration
    # one: only CUDA-family platforms ever linked the barex transport. Its
    # sources pull CUDA-only torch headers (ATen/cuda/CUDAContextLight.h ->
    # cuda_runtime_api.h), which do not resolve under the ROCm toolchain and
    # were never built on any non-CUDA lane — AMD/PPU/others keep the
    # no-impl stub, matching the arch-select this open-sourcing replaced.
    # Two flat aliases instead of one nested select: each select has a
    # single non-default key, so no multi-match can occur.
    native.alias(
        name = "rdma_transport_arch_select_impl",
        actual = select({
            "@rtp_llm//:using_rdma": ":rdma_transport_rdma_platform_impl",
            "//conditions:default": "@rtp_llm//rtp_llm/cpp/rdma_transport:rdma_transport_no_impl",
        }),
        visibility = ["//visibility:public"],
    )
    native.alias(
        name = "rdma_transport_rdma_platform_impl",
        actual = select({
            "@rtp_llm//:cuda_pre_12_9": "@rtp_llm//rtp_llm/cpp/rdma_transport/barex:barex_rdma_transport_impl",
            "@rtp_llm//:using_cu12_9_or_13_x86": "@rtp_llm//rtp_llm/cpp/rdma_transport/barex:barex_rdma_transport_impl",
            "@rtp_llm//:using_cuda12_arm": "@rtp_llm//rtp_llm/cpp/rdma_transport/barex:barex_rdma_transport_impl",
            "//conditions:default": "@rtp_llm//rtp_llm/cpp/rdma_transport:rdma_transport_no_impl",
        }),
        visibility = ["//visibility:public"],
    )

def transfer_backend_deps():
    # Same using_rdma selection as cache_store_deps(): OFF (every public
    # config) resolves to the TCP base backend; internal overlay builds set
    # --define=use_rdma=true and select the RDMA backend factory, which links
    # the closed accl-barex library via @alibaba_rdma.
    native.alias(
        name = "transfer_backend_arch_select_impl",
        actual = select({
            "@rtp_llm//:using_rdma": "@rtp_llm//rtp_llm/cpp/cache/connector/p2p/transfer:transfer_backend_rdma_impl",
            "//conditions:default": "@rtp_llm//rtp_llm/cpp/cache/connector/p2p/transfer:transfer_backend_base_impl",
        }),
    )
