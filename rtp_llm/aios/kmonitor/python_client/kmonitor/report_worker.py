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
from rtp_llm.aios.kmonitor.python_client.kmonitor.utils.hippo_helper import HippoHelper

_ReportWorker__REPORT_PORT = 4141
_ReportWorker__FLUME_CLIENT_TIMEOUT_MS = 1000

_ReportWorker__REPORT_INTERVAL_SECOND = 1
_ReportWorker__REPORT_HEADERS = {"topic": "py_kmonitor"}
_ReportWorker__REPORT_KMONITOR_MULTI_SEP = "@"
_ReportWorker__REPORT_KMONITOR_KEYVALUE_SEP = "^"


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _defer_transport_for_scr() -> bool:
    enabled = _env_enabled("RTPLLM_ENABLE_SCR")
    return enabled and os.environ.get("SCR_PHASE", "").strip().lower() in {
        "checkpoint",
        "restore",
    }


class ReportWorker(object):
    def __init__(self, *args):
        super(ReportWorker, self).__init__(*args)
        self._runtime_tags = HippoHelper.get_hippo_tags()
        self._runtime_tag_keys: set[str] = set()
        self.init_tags = self._runtime_tags.copy()
        self.init_tags.update(self.parse_kmon_tags(os.environ.get("kmonitorTags", "")))
        logging.info(
            f"kmonitor report default tags: {json.dumps(self.init_tags, indent=4)}"
        )
        self.metrics: Dict[str, MetricBase] = {}
        self.metric_lock: Lock = Lock()
        self.started = False
        self._report_thread: Thread | None = None
        self.flume = None
        self._transport_deferred = _defer_transport_for_scr()
        if self._transport_deferred:
            logging.info("defer kmonitor transport until SCR steady-point returns")
        elif HippoHelper.is_hippo_env():
            self._activate_hippo_transport()
        else:
            self.start()
            logging.info("test mode, kmonitor metrics not reported.")

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
        report_tags = data_point.tags
        if self._runtime_tag_keys:
            # Registered metrics retain seed tags. Replace only runtime identity,
            # including tags removed during fixup; ordinary startup is unaffected.
            report_tags = {
                k: v for k, v in report_tags.items() if k not in self._runtime_tag_keys
            }
            report_tags.update(self._runtime_tags)
        tag_str: str = " ".join(
            ["=".join([k, v]) for (k, v) in list(report_tags.items())]
        )
        report_message: bytes = " ".join(
            [metric_name, str(timestamp), value_str, tag_str]
        ).encode("utf-8")
        return ThriftFlumeEvent(_ReportWorker__REPORT_HEADERS, report_message)

    def get_report_events(self) -> List[ThriftFlumeEvent]:
        events: List[ThriftFlumeEvent] = []
        timestamp: int = int(round(time.time()))
        with self.metric_lock:
            for metric_name, metric in self.metrics.items():
                reported_data = metric.fetch_reported_data()
                for data_point in reported_data:
                    event = self.render_event(metric_name, timestamp, data_point)
                    events.append(event)
        return events

    def do_report(self) -> None:
        events = self.get_report_events()
        # logging.debug(f'kmonitor collected {len(events)} events.')
        if self.flume:
            self.flume.send_batch(events)
        else:
            for event in events:
                pass
                # logging.debug(event.body)

    def report_cycle(self) -> None:
        try:
            while self.started:
                time.sleep(_ReportWorker__REPORT_INTERVAL_SECOND)
                self.do_report()
        except Exception as e:
            logging.error(f"kmonitor report thread error: {e} {traceback.format_exc()}")
        logging.warn("kmonitor report process exited.")

    def start(self) -> None:
        if self._report_thread is not None and self._report_thread.is_alive():
            self.started = True
            return
        self.started = True
        report_thread = Thread(target=self.report_cycle)
        report_thread.daemon = True
        self._report_thread = report_thread
        report_thread.start()

    def stop(self) -> None:
        self.started = False

    def _activate_hippo_transport(self) -> None:
        report_host = os.environ.get("HIPPO_SLAVE_IP", "localhost")
        self.flume = FlumeClient(
            report_host,
            _ReportWorker__REPORT_PORT,
            timeout=_ReportWorker__FLUME_CLIENT_TIMEOUT_MS,
        )
        self._transport_deferred = False
        self.start()
        logging.info(
            "hippo role [%s] at host [%s-%s] started reporting kmonitor",
            HippoHelper.role,
            HippoHelper.host_ip,
            HippoHelper.container_ip,
        )

    def start_after_restore(self) -> None:
        """Start the deferred reporter once, after process identity is refreshed."""
        if not self._transport_deferred:
            return
        # The shared runtime fixup has already refreshed HippoHelper. Preserve
        # custom defaults and the dict referenced by existing KMonitor instances.
        runtime_tags = HippoHelper.get_hippo_tags()
        self._runtime_tag_keys = self._runtime_tags.keys() | runtime_tags.keys()
        self._runtime_tags = runtime_tags
        for key in self._runtime_tag_keys:
            self.init_tags.pop(key, None)
        self.init_tags.update(runtime_tags)
        if HippoHelper.is_hippo_env():
            self._activate_hippo_transport()
        else:
            self._transport_deferred = False
            self.start()


report_worker = ReportWorker()
