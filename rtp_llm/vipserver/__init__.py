"""Lazy public facade for the process-global VIP client.

Importing a helper module such as ``vipserver_proxy`` must not start network
discovery as a package-import side effect. The global client is initialized on
the first actual lookup instead.
"""


def get_host_list_by_domain_now(domain: str):
    from .vip_client import get_host_list_by_domain_now as resolve

    return resolve(domain)


def get_host_list_by_domain(domain: str):
    from .vip_client import get_host_list_by_domain as resolve

    return resolve(domain)


def get_one_validate_host_now(domain: str):
    from .vip_client import get_one_validate_host_now as resolve

    return resolve(domain)


def get_one_validate_host(domain: str):
    from .vip_client import get_one_validate_host as resolve

    return resolve(domain)
