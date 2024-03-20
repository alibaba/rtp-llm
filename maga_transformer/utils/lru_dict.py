from lru import LRU
from typing import Any, Iterator, Tuple

# lru包是cpython.so，代码提示和跳转会有问题，所以包一层增加可读性
class LruDict(object):
    def __init__(self):
        # lru_cache的大小不重要
        self._dict = LRU(100000)

    def __getitem__(self, key: str) -> Any:
        return self._dict[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._dict[key] = value

    def keys(self) -> Iterator[Any]:
        return self._dict.keys()
    
    def values(self) -> Iterator[Any]:
        return self._dict.values()

    def empty(self) -> bool:
        return len(self._dict) == 0
    
    def pop(self, key: str):
        return self._dict.pop(key)

    def poplast(self) -> Tuple[Any, Any]:
        return self._dict.popitem()
    
    def len(self):
        return len(self._dict)
    
    def items(self) -> Iterator[Tuple[Any, Any]]:
        return self._dict.items()
    
    def __contains__(self, key: Any) -> bool:
        return key in self._dict
