import logging
import threading
import traceback

from rtp_llm.vipserver.host import Host
from rtp_llm.vipserver.netutil import NetUtils
from rtp_llm.vipserver.update_thread import UpdateThread
from rtp_llm.vipserver.vipserver_proxy import VIPServerProxy


class HostReactor:
    domain_map = {}
    domain_update_lock = threading.Lock()

    def __init__(self, proxy: VIPServerProxy):
        self.domain_map: dict[str, list[Host]] = {}
        self.domain_update_lock = threading.Lock()
        self.proxy = proxy
        self.update_domain_thread = UpdateThread(
            "vipserver-domain-update", self.refresh_cache_domain_srv_lst
        )
        self.started = False

    def start(self):
        if not self.proxy.started:
            self.proxy.start()
        if not self.started:
            self.update_domain_thread.start()
            logging.info(
                f"vipserver domain update thread started. to refresh domains: {self.refresh_cache_domain_srv_lst}"
            )
            self.started = True

    def close(self):
        if self.started:
            self.update_domain_thread.stop_flag = True
            self.update_domain_thread.join()
            self.started = False
            logging.info(
                f"vipserver domain update thread stopped. to refresh domains: {self.refresh_cache_domain_srv_lst}"
            )

        self.update_domain_thread.join()
        if self.proxy.started:
            self.proxy.close()

    def update_domain_map(self, new_map: dict[str, list[Host]]):
        with self.domain_update_lock:
            for k, v in new_map.items():
                if v:
                    self.domain_map[k] = v
                else:
                    logging.warning(
                        "%s returned an empty host list; retaining cached hosts", k
                    )

    def refresh_domain_srv_lst(self, domain: str):
        """
        refresh host list of target vip domain right now
        :param domain:
        :return:
        """
        try:
            params = {
                "dom": domain,
                "qps": 0,
                "clientIP": NetUtils.get_ip_addr(),
                "udpPort": 55963,
                "encoding": "GBK",
            }
            resp_json = self.proxy.req_api("srvIPXT", params)
            if resp_json is None:
                return
            if "hosts" in resp_json:
                hosts = []
                for host in resp_json["hosts"]:
                    if host["valid"]:
                        hosts.append(Host(host["ip"], host["port"]))
                self.update_domain_map({domain: hosts})
            else:
                self.update_domain_map({domain: []})

        except Exception as e:
            logging.error(f"{domain} failed to refresh vipserver domain server list", e)
            stack_summary = traceback.format_exception(type(e), e, e.__traceback__)
            stack_str = "\n".join(stack_summary)
            logging.error(f"error stack: {stack_str}")

    def refresh_cache_domain_srv_lst(self):
        """
        refresh host list for each domain right now
        :return:
        """
        for k, v in self.domain_map.items():
            self.refresh_domain_srv_lst(k)

    def get_host_list_by_domain_now(self, domain: str):
        """
        get available host list by domain without cache, will req vipserver api right now
        :param domain: vipserver domain
        :return: host list
        """
        self.refresh_domain_srv_lst(domain)
        return self.domain_map.get(domain)

    def get_host_list_by_domain(self, domain: str):
        """
        get available host list by domain with cache
        :param domain: vipserver domain
        :return: host list
        """
        hosts = self.domain_map.get(domain)
        if hosts is None:
            return self.get_host_list_by_domain_now(domain)
        return hosts
