# Public stub of the @internal_impls seam: open defaults; the closed checkout
# swaps the repository via --override_repository (see .internal_bazelrc there).

EMBEDDING_ARPC_IMPL = "@rtp_llm//rtp_llm/cpp/embedding_engine:embedding_engine_arpc_server_impl"

# Arch branches that carry the impl keep the open service in open builds.
SUBSCRIBE_IMPL = "@rtp_llm//rtp_llm/cpp/disaggregate/load_balancer/subscribe:subscribe_service_impl"

def extension_packages():
    return ["rtp_llm"]

def extension_package_deps():
    return []

def extension_frontend_package_deps():
    return []

def extension_lib_deps():
    return []

def extension_frontend_lib_deps():
    return []

def internal_only_deps():
    pass
