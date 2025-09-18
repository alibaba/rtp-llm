import logging
import threading
import traceback

from rtp_llm.vipserver.host import Host
from rtp_llm.vipserver.netutil import NetUtils
from rtp_llm.vipserver.update_thread import UpdateThread
from rtp_llm.vipserver.vipserver_proxy import VIPServerProxy

DOMAIN_FAILED_CNT_THRESHOLD = 20


class HostReactor:
    domain_map = {}
    domain_update_lock = threading.Lock()

    def __init__(self, proxy: VIPServerProxy):
        self.domain_map: dict[str, list[Host]] = {}
        self.domain_failed_cnt: dict[str, int] = {}
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
        self.domain_update_lock.acquire()
        for k, v in new_map.items():
            if v:
                self.domain_map[k] = v
                self.domain_failed_cnt[k] = 0
            else:
                self.domain_failed_cnt[k] = self.domain_failed_cnt.get(k, 0) + 1
                logging.warning(
                    f"{k} failed to refresh vipserver domain server list: empyt host list - {self.domain_failed_cnt[k]} times"
                )
                if self.domain_failed_cnt[k] >= DOMAIN_FAILED_CNT_THRESHOLD:
                    logging.warning(
                        f"{k} has failed {self.domain_failed_cnt[k]} times, set server list to empty."
                    )
                    self.domain_map[k] = []
                    self.domain_failed_cnt[k] = 0

        self.domain_update_lock.release()

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
            if resp_json and "hosts" in resp_json:
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
