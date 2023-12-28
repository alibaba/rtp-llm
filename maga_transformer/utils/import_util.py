import importlib.util

def load_module(module_path: str):    
    module_spec = importlib.util.spec_from_file_location('inference_module', module_path)
    if module_spec is None:
        raise ModuleNotFoundError(f'failed to load module from [{module_path}]')

    imported_module = importlib.util.module_from_spec(module_spec)

    if module_spec.loader != None:
        module_spec.loader.exec_module(imported_module)
    else:
        raise Exception(f"ModuleSpec [{module_spec}] has no loader.")
    return imported_module
