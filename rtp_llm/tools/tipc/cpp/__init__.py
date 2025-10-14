# tipc/__init__.py

try:
    from ._tipc_lib import import_tensor_ipc, export_tensor_ipc 
    
except ImportError as e:
    raise ImportError