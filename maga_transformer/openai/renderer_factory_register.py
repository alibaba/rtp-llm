from typing import Dict, Any, Type

_renderer_factory: Dict[str, Type[Any]] = {}

def register_renderer(name: str, renderer_type: Any):
    global _renderer_factory
    if name in _renderer_factory and _renderer_factory[name] != renderer_type:
        raise Exception(f"try register renderer {name} with type {_renderer_factory[name]} and {renderer_type}, confict!")
    _renderer_factory[name] = renderer_type