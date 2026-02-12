from typing import Any, Dict, List, Type, Union

_multimodal_mixin_factory: Dict[str, Type[Any]] = {}


def register_multimodal_mixin(name: Union[str, List[str]], multimodal_mixin: Any):
    global _multimodal_mixin_factory
    if isinstance(name, List):
        for n in name:
            register_multimodal_mixin(n, multimodal_mixin)
    else:
        if (
            name in _multimodal_mixin_factory
            and _multimodal_mixin_factory[name] != multimodal_mixin
        ):
            raise Exception(
                f"try register model {name} with type {_multimodal_mixin_factory[name]} and {multimodal_mixin}, confict!"
            )
        _multimodal_mixin_factory[name] = multimodal_mixin


def get_multimodal_mixin_cls(name: str) -> Type[Any]:
    if name not in _multimodal_mixin_factory:
        raise ValueError(f"Multimodal mixin {name} not found")
    return _multimodal_mixin_factory[name]
