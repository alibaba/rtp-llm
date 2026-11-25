# Loading proxy/reactor types must not start the default network client.
__all__ = [
    "get_host_list_by_domain",
    "get_host_list_by_domain_now",
    "get_one_validate_host",
    "get_one_validate_host_now",
]


def __getattr__(name):
    if name in __all__:
        from . import vip_client

        return getattr(vip_client, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
