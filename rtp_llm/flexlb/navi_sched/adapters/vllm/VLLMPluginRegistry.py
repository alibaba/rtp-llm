from collections.abc import Callable
from typing import Any

_engine_observer_factory: Callable[[Any, Any, int], Any] | None = None
_engine_transport_factory: Callable[[Any, Any], Any] | None = None
_renderer_transport_factory: Callable[[Any], Any] | None = None
_frontend_request_interceptor: Callable[[Any, Any], bool] | None = None
_frontend_app_configurator: Callable[[Any], None] | None = None
_scheduler_iteration_metrics_enabled = False


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


def scheduler_iteration_metrics_enabled() -> bool:
    return _scheduler_iteration_metrics_enabled


def install(plugins):
    import sys

    adapter = sys.modules[__name__]
    existing = plugins.get_engine_adapter()
    if existing is not None and existing is not adapter:
        raise RuntimeError("an engine adapter is already registered")
    plugins.register_engine_adapter(adapter)


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
