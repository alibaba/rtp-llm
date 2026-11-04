import os


def _intercept_frontend_request(unused_request, unused_raw_request) -> bool:
    return True


def register_enabled_vllm_plugin() -> None:
    hash_seed = os.environ.setdefault("PYTHONHASHSEED", "0")
    if hash_seed != "0":
        raise ValueError("NaviSched requires PYTHONHASHSEED=0")

    from vllm.plugins import (
        register_engine_observer_factory,
        register_engine_transport_factory,
        register_frontend_request_interceptor,
        register_frontend_app_configurator,
        register_renderer_transport_factory,
        register_scheduler_iteration_metrics,
    )

    from .VLLMEngineObserver import create_vllm_engine_observer
    from .VLLMEngineTransport import create_vllm_engine_transport
    from .VLLMDeferredRenderer import create_deferred_renderer
    from .VLLMNativePreparation import configure_native_preparation

    register_engine_observer_factory(create_vllm_engine_observer)
    register_engine_transport_factory(create_vllm_engine_transport)
    register_renderer_transport_factory(create_deferred_renderer)
    register_frontend_request_interceptor(_intercept_frontend_request)
    register_frontend_app_configurator(configure_native_preparation)
    register_scheduler_iteration_metrics()
