"""Expose the platform-selected PyTorch wheel's pybind11 headers without linking its runtime."""

def _headers_impl(ctx):
    info = cc_common.merge_cc_infos(cc_infos = [dep[CcInfo] for dep in ctx.attr.deps])
    return [CcInfo(compilation_context = info.compilation_context)]

binding_headers = rule(
    implementation = _headers_impl,
    attrs = {"deps": attr.label_list(providers = [CcInfo])},
)
