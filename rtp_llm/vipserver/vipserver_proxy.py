import logging
import os
import random
import threading

import requests

from rtp_llm.vipserver.label_collector import get_environments
from rtp_llm.vipserver.netutil import NetUtils
from rtp_llm.vipserver.update_thread import UpdateThread

_ADDRESS_SERVER_HTTP_TIMEOUT_S = (1.0, 5.0)
_VIPSERVER_API_HTTP_TIMEOUT_S = (1.0, 3.0)


def get_address_server_params():
    environments = get_environments()
    labels = ""
    for k, v in environments.items():
        labels += f"{k}:{v},"
    return {"labels": labels}


def get_query_string(params: dict) -> str:
    query_string = ""
    for k, v in params.items():
        query_string += f"{k}={v}&"
    return query_string


class VIPServerProxy:
    srv_hosts = []
    srv_update_lock = threading.Lock()

    def __init__(self):
        self.jmenv = "jmenv.tbsite.net:8080"
        jm_domain = os.getenv("address.server.domain")
        if jm_domain:
            jm_port = os.getenv("address.server.port")
            self.jmenv = (
                jm_domain.strip() + ":" + (jm_port.strip() if jm_port else "8080")
            )
        jm_domain = os.getenv("com.alibaba.vipserver.jmenv")
        if jm_domain:
            self.jmenv = jm_domain

        self.update_srv_thread = UpdateThread(
            "vipserver-srv-update", self.refresh_srv_lst
        )
        self.started = False

    def start(self):
        # load srv list
        self.refresh_srv_lst()
        self.update_srv_thread.start()
        self.started = True

    def close(self):
        self.update_srv_thread.stop_flag = True
        self.update_srv_thread.join()
        self.started = False

    def refresh_srv_lst(self):
        """
        refresh vipserver server list right now
        :return:
        """
        params = get_address_server_params()
        query_string = get_query_string(params)

        jmenv_url = f"http://{self.jmenv}/vipserver/serverlist?nofix=1&{query_string}"

        try:
            resp = requests.get(jmenv_url, timeout=_ADDRESS_SERVER_HTTP_TIMEOUT_S).text
            srv_lst = []
            for srv in resp.split("\n"):
                if srv.strip() == "":
                    continue
                if not NetUtils.is_valid_ipv4(srv):
                    print(f"found invalid vipserver ip: {srv}, skip")
                    continue
                srv_lst.append(srv)

            random.shuffle(srv_lst)
            with self.srv_update_lock:
                self.srv_hosts = srv_lst
        except Exception as e:
            logging.error(
                f"failed to refresh vipserver server list, exception : {str(e)}"
            )

    def req_api(self, api: str, params: dict[str, str]):
        """
        req vipserver api
        :param api: api name
        :param params: params map
        :return: response json
        """
        with self.srv_update_lock:
            srv_snapshot = tuple(self.srv_hosts)

        if not srv_snapshot:
            logging.error("no vipserver hosts available")
            return None

        req_params = dict()
        if params:
            req_params.update(params)
            req_params.update(get_address_server_params())

        last_error: Exception | None = None
        for srv in srv_snapshot:
            resp = None
            try:
                resp = requests.get(
                    f"http://{srv}:80/vipserver/api/{api}?{get_query_string(req_params)}",
                    timeout=_VIPSERVER_API_HTTP_TIMEOUT_S,
                )
                return resp.json()
            except Exception as e:
                body = ""
                try:
                    body = resp.text[:200] if resp is not None else ""
                except Exception:
                    pass
                logging.info(
                    "req api from vipserver failed, api=%s srv=%s params=%s "
                    "error=%s body=%s",
                    api,
                    srv,
                    params,
                    e,
                    body,
                )
                last_error = e

        logging.error("all vipserver hosts failed for api %s: %s", api, last_error)
        return None
