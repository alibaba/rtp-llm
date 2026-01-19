from typing import Any, Dict, List, Type, Union

_tokenizer_factory: Dict[str, Type[Any]] = {}


def register_tokenizer(name: Union[str, List[str]], tokenizer: Any):
    global _tokenizer_factory
    if isinstance(name, List):
        for n in name:
            register_tokenizer(n, tokenizer)
    else:
        if name in _tokenizer_factory and _tokenizer_factory[name] != tokenizer:
            raise Exception(
                f"try register model {name} with type {_tokenizer_factory[name]} and {tokenizer}, confict!"
            )
        _tokenizer_factory[name] = tokenizer
