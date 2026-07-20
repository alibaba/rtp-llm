import logging
import threading

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
        with self.domain_update_lock:
            for k, v in new_map.items():
                if v:
                    self.domain_map[k] = v
                    self.domain_failed_cnt[k] = 0
                else:
                    failed_count = self.domain_failed_cnt.get(k, 0) + 1
                    self.domain_failed_cnt[k] = failed_count
                    logging.warning(
                        "%s failed to refresh vipserver domain server list: "
                        "empty host list - %s times",
                        k,
                        failed_count,
                    )
                    # Keep a stale healthy snapshot while refreshes fail. Cold
                    # failures remain absent from domain_map so the discovery
                    # layer's short negative-cache TTL can trigger a foreground
                    # retry. domain_failed_cnt still registers the domain for
                    # background refresh below.
                    if failed_count >= DOMAIN_FAILED_CNT_THRESHOLD:
                        logging.warning(
                            "%s has failed %s times, set server list to empty",
                            k,
                            failed_count,
                        )
                        self.domain_map[k] = []
                        self.domain_failed_cnt[k] = 0

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

        except Exception:
            logging.exception(
                "%s failed to refresh vipserver domain server list", domain
            )
            self.update_domain_map({domain: []})

    def refresh_cache_domain_srv_lst(self):
        """
        refresh host list for each domain right now
        :return:
        """
        with self.domain_update_lock:
            domains = tuple(self.domain_map.keys() | self.domain_failed_cnt.keys())
        for domain in domains:
            self.refresh_domain_srv_lst(domain)

    def get_host_list_by_domain_now(self, domain: str):
        """
        get available host list by domain without cache, will req vipserver api right now
        :param domain: vipserver domain
        :return: host list
        """
        self.refresh_domain_srv_lst(domain)
        with self.domain_update_lock:
            return self.domain_map.get(domain)

    def get_host_list_by_domain(self, domain: str):
        """
        get available host list by domain with cache
        :param domain: vipserver domain
        :return: host list
        """
        hosts = self.get_host_list_by_domain_cached(domain)
        if not hosts:
            return self.get_host_list_by_domain_now(domain)
        return hosts

    def get_host_list_by_domain_cached(self, domain: str):
        """Return the current snapshot without triggering network I/O."""
        # Writers replace whole list objects under ``domain_update_lock`` and
        # never mutate a published list. A CPython dict get therefore yields a
        # safe snapshot reference without making the grpc.aio hot path wait on
        # a background refresh thread's lock or logging.
        return self.domain_map.get(domain)
