from collections.abc import Callable
from typing import Any

_engine_observer_factory: Callable[[Any, Any, int], Any] | None = None
_engine_transport_factory: Callable[[Any, Any], Any] | None = None
_renderer_transport_factory: Callable[[Any], Any] | None = None
_frontend_request_interceptor: Callable[[Any, Any], bool] | None = None
_frontend_app_configurator: Callable[[Any], None] | None = None
_scheduler_iteration_metrics_enabled = False
_worker_observer_factory: Callable[[Any, Any], Any] | None = None
_legacy_metrics_registrar = None


def register_engine_observer_factory(factory: Callable[[Any, Any, int], Any]) -> None:
    global _engine_observer_factory
    if _engine_observer_factory is not None and _engine_observer_factory is not factory:
        raise RuntimeError("an engine observer factory is already registered")
    _engine_observer_factory = factory


def create_engine_observer(
    vllm_config: Any,
    kv_cache_config: Any,
    max_concurrent_batches: int,
) -> Any | None:
    if _engine_observer_factory is None:
        return None
    return _engine_observer_factory(
        vllm_config,
        kv_cache_config,
        max_concurrent_batches,
    )


def register_worker_observer_factory(factory: Callable[[Any, Any], Any]) -> None:
    global _worker_observer_factory
    if _worker_observer_factory is not None and _worker_observer_factory is not factory:
        raise RuntimeError("a worker observer factory is already registered")
    _worker_observer_factory = factory


def create_worker_observer(vllm_config: Any, worker: Any) -> Any:
    if _worker_observer_factory is None:
        return worker
    return _worker_observer_factory(vllm_config, worker)


def register_engine_transport_factory(factory: Callable[[Any, Any], Any]) -> None:
    global _engine_transport_factory
    if (
        _engine_transport_factory is not None
        and _engine_transport_factory is not factory
    ):
        raise RuntimeError("an engine transport factory is already registered")
    _engine_transport_factory = factory


def create_engine_transport(vllm_config: Any, engine_core: Any) -> Any:
    if _engine_transport_factory is None:
        return engine_core
    return _engine_transport_factory(vllm_config, engine_core)


def register_renderer_transport_factory(factory: Callable[[Any], Any]) -> None:
    global _renderer_transport_factory
    if (
        _renderer_transport_factory is not None
        and _renderer_transport_factory is not factory
    ):
        raise RuntimeError("a renderer transport factory is already registered")
    _renderer_transport_factory = factory


def create_renderer_transport(renderer: Any) -> Any:
    if _renderer_transport_factory is None:
        return renderer
    return _renderer_transport_factory(renderer)


def register_frontend_request_interceptor(
    interceptor: Callable[[Any, Any], bool],
) -> None:
    global _frontend_request_interceptor
    if (
        _frontend_request_interceptor is not None
        and _frontend_request_interceptor is not interceptor
    ):
        raise RuntimeError("a frontend request interceptor is already registered")
    _frontend_request_interceptor = interceptor


def is_frontend_request_intercepted(request: Any, raw_request: Any) -> bool:
    if _frontend_request_interceptor is None:
        return False
    return bool(_frontend_request_interceptor(request, raw_request))


def register_frontend_app_configurator(configurator: Callable[[Any], None]) -> None:
    global _frontend_app_configurator
    if (
        _frontend_app_configurator is not None
        and _frontend_app_configurator is not configurator
    ):
        raise RuntimeError("a frontend app configurator is already registered")
    _frontend_app_configurator = configurator


def configure_frontend_app(app: Any) -> None:
    if _frontend_app_configurator is not None:
        _frontend_app_configurator(app)


def register_scheduler_iteration_metrics() -> None:
    global _scheduler_iteration_metrics_enabled
    _scheduler_iteration_metrics_enabled = True
    if _legacy_metrics_registrar is not None:
        _legacy_metrics_registrar()


def scheduler_iteration_metrics_enabled() -> bool:
    return _scheduler_iteration_metrics_enabled


def install(plugins):
    import sys

    global _legacy_metrics_registrar
    adapter = sys.modules[__name__]
    get_adapter = getattr(plugins, "get_engine_adapter", None)
    register_adapter = getattr(plugins, "register_engine_adapter", None)
    if callable(get_adapter) and callable(register_adapter):
        existing = get_adapter()
        if existing is not None and existing is not adapter:
            raise RuntimeError("an engine adapter is already registered")
        register_adapter(adapter)
        _legacy_metrics_registrar = None
        return

    # Older vLLM delegates through factory hooks. Keep callback ownership here.
    required = ("register_engine_observer_factory", "register_engine_transport_factory")
    if any(not callable(getattr(plugins, name, None)) for name in required):
        raise RuntimeError("vLLM exposes neither an engine adapter nor legacy factory hooks")
    factories = (
        ("register_engine_observer_factory", create_engine_observer),
        ("register_engine_transport_factory", create_engine_transport),
        ("register_worker_observer_factory", create_worker_observer),
        ("register_renderer_transport_factory", create_renderer_transport),
        ("register_frontend_request_interceptor", is_frontend_request_intercepted),
        ("register_frontend_app_configurator", configure_frontend_app),
    )
    for name, factory in factories:
        register = getattr(plugins, name, None)
        if callable(register):
            register(factory)
    metrics = getattr(plugins, "register_scheduler_iteration_metrics", None)
    _legacy_metrics_registrar = metrics if callable(metrics) else None


def raise_if_error(finish_reason, request_id, stop_reason, logger):
    from .VLLMGenerationError import raise_if_error as check_generation_error

    return check_generation_error(finish_reason, request_id, stop_reason, logger)


def is_rate_limited(error):
    from http import HTTPStatus

    from vllm.entrypoints.openai.engine.protocol import GenerationError

    return (
        isinstance(error, GenerationError)
        and error.status_code == HTTPStatus.TOO_MANY_REQUESTS
    )
