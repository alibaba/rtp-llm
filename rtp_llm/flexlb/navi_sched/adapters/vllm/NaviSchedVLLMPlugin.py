import os


def _intercept_frontend_request(unused_request, unused_raw_request) -> bool:
    return True


def register_enabled_vllm_plugin() -> None:
    hash_seed = os.environ.setdefault("PYTHONHASHSEED", "0")
    if hash_seed != "0":
        raise ValueError("NaviSched requires PYTHONHASHSEED=0")

    from vllm import plugins
    from vllm.plugins import (
        register_engine_observer_factory,
        register_engine_transport_factory,
    )

    from .VLLMDeferredRenderer import create_deferred_renderer
    from .VLLMEngineObserver import create_vllm_engine_observer
    from .VLLMEngineTransport import create_vllm_engine_transport
    from .VLLMNativePreparation import configure_native_preparation

    register_engine_observer_factory(create_vllm_engine_observer)
    register_worker = getattr(plugins, "register_worker_observer_factory", None)
    if register_worker is not None:
        from .VLLMWorkerObserver import create_vllm_worker_observer

        register_worker(create_vllm_worker_observer)
    # DashLLM keeps its existing PULL transport by default. An explicit
    # EngineCore redirect opt-in allows the AsyncLLM frontend to route the
    # already-constructed vLLM request without changing DashServing's wire
    # protocol or DashLLM's request preparation and P/D orchestration.
    dashllm_prefill = os.environ.get(
        "TPU_RAIDEN_JOB_NAME", ""
    ).strip().lower() == "prefill" and (
        os.environ.get("VLLM_NAVI_SCHED_NATIVE_INGRESS") == "1"
        or os.environ.get("DS_ENABLE_DSERV") == "1"
    )
    redirect_prefill = os.environ.get("VLLM_NAVI_SCHED_PREFILL_CORE_REDIRECT") == "1"
    if not dashllm_prefill or redirect_prefill:
        register_engine_transport_factory(create_vllm_engine_transport)
    register_frontend = getattr(plugins, "register_frontend_request_interceptor", None)
    if register_frontend is not None:
        register_frontend(_intercept_frontend_request)
    register_metrics = getattr(plugins, "register_scheduler_iteration_metrics", None)
    if register_metrics is not None:
        register_metrics()
    if not dashllm_prefill:
        plugins.register_renderer_transport_factory(create_deferred_renderer)
        plugins.register_frontend_app_configurator(configure_native_preparation)
