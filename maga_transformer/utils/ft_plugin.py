import os
import logging
from typing import Any, List, Protocol, Callable, Optional, Tuple

from maga_transformer.utils.import_util import load_module

# here declare the expected format for plugin, **kwargs used for back compatibility
class ModifyPromptCallable(Protocol):
    def __call__(self, prompt: str, **kwargs: Any) -> str:
        ...

class MultiModalModifyPromptCallable(Protocol):
    def __call__(self, prompt: str, urls: List[str], mm_token: str, **kwargs: Any) -> Tuple[str, List[Any]]:
        ...

class EncodeCallable(Protocol):
    def __call__(self, prompt: str, **kwargs: Any) -> List[int]:
        ...

class DecodeCallable(Protocol):
    def __call__(self, tokens: List[int], **kwargs: Any) -> str:
        ...

class ModifyResponseCallable(Protocol):
    def __call__(self, response: str, **kwargs: Any) -> str:
      ...

class StopGenerateCallable(Protocol):
    def __call__(self, resposne: str, **kwargs: Any) -> bool:
      ...

class Plugin(object):
    def __init__(self, plugin_cls: Optional[type] = None):
        self._plugin = plugin_cls() if plugin_cls is not None else None
        self.modify_prompt_plugin: Optional[ModifyPromptCallable] = self.get_if_exists(self._plugin, 'modify_prompt_plugin')
        self.process_encode_plugin: Optional[EncodeCallable] = self.get_if_exists(self._plugin, 'process_encode_plugin')
        self.process_decode_plugin: Optional[DecodeCallable] = self.get_if_exists(self._plugin, 'process_decode_plugin')
        self.modify_response_plugin: Optional[ModifyResponseCallable] = self.get_if_exists(self._plugin, 'modify_response_plugin')
        self.stop_generate_plugin: Optional[StopGenerateCallable] = self.get_if_exists(self._plugin, 'stop_generate_plugin')

    def get_if_exists(self, instance: Any, plugin_name: str) -> Optional[Callable[..., Any]]:
        return getattr(instance, plugin_name, None) if instance is not None else None

class PluginLoader(object):
    def __init__(self):
        self.reload()

    def get_if_exists(self, module: Any, plugin_name: str) -> Callable[..., Any]:
        if module is None:
            return None
        return getattr(module, plugin_name, None)

    def reload(self):
        self.plugin_cls = None
        plugin_path = os.environ.get('FT_PLUGIN_PATH', None)
        if plugin_path is not None:
            try:
                module = load_module(plugin_path)
                self.plugin_cls = module.CustomPlugin
            except Exception as e:
                logging.info("failed to load plugin: " + str(e))

    def get_plugin(self):
        # 每次调用pipeline作为生命周期生成一个instance
        return Plugin(self.plugin_cls)

plguin_loader = PluginLoader()
