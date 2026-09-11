"""Compile the transport bridge without embedding a second model runtime."""

def _model_headers_impl(ctx):
    return [CcInfo(compilation_context = ctx.attr.models[CcInfo].compilation_context)]

model_headers = rule(
    implementation = _model_headers_impl,
    attrs = {"models": attr.label(mandatory = True, providers = [CcInfo])},
)
