import random

from rtp_llm.vipserver.host_reactor import HostReactor
from rtp_llm.vipserver.vipserver_proxy import VIPServerProxy


class VipClient:
    def __init__(self, host_reactor: HostReactor):
        self.host_reactor = host_reactor
        if not self.host_reactor.started:
            self.host_reactor.start()

    def get_host_list_by_domain_now(self, domain: str):
        """
        get available host list by domain without cache, will req vipserver api right now
        :param domain: vipserver domain
        :return: host list
        """
        return self.host_reactor.get_host_list_by_domain_now(domain)

    def get_host_list_by_domain(self, domain: str):
        """
        get available host list by domain with cache
        :param domain: vipserver domain
        :return: host list
        """
        return self.host_reactor.get_host_list_by_domain(domain)

    def get_host_list_by_domain_cached(self, domain: str):
        """Return the current host snapshot without triggering a refresh."""
        return self.host_reactor.get_host_list_by_domain_cached(domain)

    def get_one_validate_host_now(self, domain: str):
        """
        choose one valid host randomly by domain without cache, will req vipserver api right now
        :param domain: vipserver domain
        :return: host list
        """
        return random.choice(self.get_host_list_by_domain_now(domain))

    def get_one_validate_host(self, domain: str):
        """
        choose one valid host randomly by domain with cache
        :param domain: vipserver domain
        :return: host list
        """
        return random.choice(self.get_host_list_by_domain(domain))

    def get_one_validate_host_cached(self, domain: str):
        """Choose from the current snapshot without triggering network I/O."""
        hosts = self.get_host_list_by_domain_cached(domain)
        return random.choice(hosts) if hosts else None


global_vip_client = VipClient(HostReactor(VIPServerProxy()))


def get_host_list_by_domain_now(domain: str):
    """
    get available host list by domain without cache, will req vipserver api right now
    :param domain: vipserver domain
    :return: host list
    """
    return global_vip_client.get_host_list_by_domain_now(domain)


def get_host_list_by_domain(domain: str):
    """
    get available host list by domain with cache
    :param domain: vipserver domain
    :return: host list
    """
    return global_vip_client.get_host_list_by_domain(domain)


def get_host_list_by_domain_cached(domain: str):
    """Return the current host snapshot without triggering a refresh."""
    return global_vip_client.get_host_list_by_domain_cached(domain)


def get_one_validate_host_now(domain: str):
    """
    choose one valid host randomly by domain without cache, will req vipserver api right now
    :param domain: vipserver domain
    :return: host list
    """
    return global_vip_client.get_one_validate_host_now(domain)


def get_one_validate_host(domain: str):
    """
    choose one valid host randomly by domain with cache
    :param domain: vipserver domain
    :return: host list
    """
    return global_vip_client.get_one_validate_host(domain)


def get_one_validate_host_cached(domain: str):
    """Choose from the current snapshot without triggering network I/O."""
    return global_vip_client.get_one_validate_host_cached(domain)
