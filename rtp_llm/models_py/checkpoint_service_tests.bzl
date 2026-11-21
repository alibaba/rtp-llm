"""Checkpoint ownership runs on CPU; C++ inference results require its driver."""

def checkpoint_service_tests(name, native_name, cpu_case, native_case, **kwargs):
    """Exercise the same service fixture in CPU and linked compute environments."""
    native.py_test(
        name = name,
        args = [cpu_case],
        **kwargs
    )
    native.py_test(
        name = native_name,
        args = [native_case],
        tags = ["L20_CU13"],
        exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
        **kwargs
    )
