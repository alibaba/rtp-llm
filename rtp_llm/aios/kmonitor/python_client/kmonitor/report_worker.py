import json
import logging
import os
import time
import traceback
from threading import Lock, Thread
from typing import Dict, List

from rtp_llm.aios.kmonitor.python_client.flume.pyflume import FlumeClient
from rtp_llm.aios.kmonitor.python_client.flume.ttypes import ThriftFlumeEvent
from rtp_llm.aios.kmonitor.python_client.kmonitor.metrics.metric_base import (
    MetricBase,
    MetricDataPoint,
)
from rtp_llm.aios.kmonitor.python_client.kmonitor.reporting import (
    reporting_epoch,
    reporting_lock,
    reporting_send_lock,
)
from rtp_llm.aios.kmonitor.python_client.kmonitor.utils.hippo_helper import HippoHelper

_ReportWorker__REPORT_HOST = os.getenv("HIPPO_SLAVE_IP", "localhost")
_ReportWorker__REPORT_PORT = 4141
_ReportWorker__FLUME_CLIENT_TIMEOUT_MS = 1000

_ReportWorker__REPORT_INTERVAL_SECOND = 1
_ReportWorker__REPORT_HEADERS = {"topic": "py_kmonitor"}
_ReportWorker__REPORT_KMONITOR_MULTI_SEP = "@"
_ReportWorker__REPORT_KMONITOR_KEYVALUE_SEP = "^"


class ReportWorker(object):
    def __init__(self, *args):
        super(ReportWorker, self).__init__(*args)
        self.init_tags = HippoHelper.get_hippo_tags()
        self.init_tags.update(self.parse_kmon_tags(os.environ.get("kmonitorTags", "")))
        logging.info(
            f"kmonitor report default tags: {json.dumps(self.init_tags, indent=4)}"
        )
        self.metrics: Dict[str, MetricBase] = {}
        self.metric_lock: Lock = Lock()
        self.send_lock: Lock = Lock()
        self.flume_enabled = HippoHelper.is_hippo_env()
        self.flume = None
        self.started = False
        # Construction follows the same pause fence as later reconnects.
        # A reporter created while sleeping stays disconnected until wake.
        with reporting_send_lock():
            self.reporting_epoch = reporting_epoch()
            if self.flume_enabled and self.reporting_epoch % 2 == 0:
                self.flume = self._connect_flume()
        self.start()
        if self.flume_enabled:
            logging.info(
                f"hippo role [{HippoHelper.role}] at host [{HippoHelper.host_ip}-{HippoHelper.container_ip}] "
                "started reporting kmonitor."
            )
        else:
            logging.info("test mode, kmonitor metrics not reported.")

    def _connect_flume(self):
        return FlumeClient(
            _ReportWorker__REPORT_HOST,
            _ReportWorker__REPORT_PORT,
            timeout=_ReportWorker__FLUME_CLIENT_TIMEOUT_MS,
        )

    def parse_kmon_tags(self, kmon_tags_str: str) -> Dict[str, str]:
        kmon_tags: Dict[str, str] = {}
        if not kmon_tags_str:
            return {}
        for tag in kmon_tags_str.split(_ReportWorker__REPORT_KMONITOR_MULTI_SEP):
            kv = tag.split(_ReportWorker__REPORT_KMONITOR_KEYVALUE_SEP)
            if len(kv) != 2:
                logging.error(f"kmon parse tags failed: tag can not split: {tag}")
                return {}
            kmon_tags[kv[0].strip()] = kv[1].strip()
        return kmon_tags

    def register_metric(self, metric: MetricBase) -> None:
        with self.metric_lock:
            if metric.name in self.metrics.keys():
                raise Exception(
                    f"metric {metric.name} already registered, can not register again."
                )
            self.metrics[metric.name] = metric

    def render_event(
        self, metric_name: str, timestamp: int, data_point: MetricDataPoint
    ) -> ThriftFlumeEvent:
        value_str = str(data_point.value)
        tag_str: str = " ".join(
            ["=".join([k, v]) for (k, v) in list(data_point.tags.items())]
        )
        report_message: bytes = " ".join(
            [metric_name, str(timestamp), value_str, tag_str]
        ).encode("utf-8")
        return ThriftFlumeEvent(_ReportWorker__REPORT_HEADERS, report_message)

    def get_report_events(self) -> List[ThriftFlumeEvent]:
        timestamp: int = int(round(time.time()))
        with reporting_lock(), self.metric_lock:
            snapshots = [
                (metric_name, metric.fetch_reported_data())
                for metric_name, metric in self.metrics.items()
            ]
        # Rendering can be much slower than taking the collector snapshots.
        # Keep it outside producer locks; do_report still owns the send fence.
        return [
            self.render_event(metric_name, timestamp, data_point)
            for metric_name, reported_data in snapshots
            for data_point in reported_data
        ]

    def do_report(self) -> None:
        # Fence local senders without holding the state/producer lock over I/O.
        # Keep draining/resetting collectors while paused, without sending even
        # an empty batch (the transport would otherwise reconnect itself).
        with reporting_send_lock(), self.send_lock:
            epoch = reporting_epoch()
            if epoch != self.reporting_epoch:
                if self.flume is not None:
                    self.flume.close()
                    self.flume = None
                self.reporting_epoch = epoch
            events = self.get_report_events()
            if epoch % 2:
                return
            if self.flume is None and self.flume_enabled:
                self.flume = self._connect_flume()
            if self.flume is not None:
                self.flume.send_batch(events)

    def report_cycle(self) -> None:
        try:
            while self.started:
                time.sleep(_ReportWorker__REPORT_INTERVAL_SECOND)
                self.do_report()
        except Exception as e:
            logging.error(f"kmonitor report thread error: {e} {traceback.format_exc()}")
        logging.warn("kmonitor report process exited.")

    def start(self) -> None:
        self.started = True
        report_thread = Thread(target=self.report_cycle)
        report_thread.daemon = True
        report_thread.start()

    def stop(self) -> None:
        self.started = False


report_worker = ReportWorker()
