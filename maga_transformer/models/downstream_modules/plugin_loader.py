import os
import sys
import logging
import importlib.util
from types import ModuleType

def load_module(file_path: str) -> ModuleType:
    if not os.path.exists(file_path):
        logging.info("file_path[%s] not exist", file_path)
        raise Exception(f"file_path[{file_path}] not exist")
    _, file_name = os.path.split(file_path)
    module_name, _ = os.path.splitext(file_name)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    logging.info('load from %s %s', module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

class UserModuleLoader(object):
    @staticmethod
    def load(module_path: str):                
        module = load_module(module_path)
        return module.UserModule