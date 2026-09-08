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
    enabled = any(
        _env_enabled(name)
        for name in ("RTPLLM_ENABLE_SCR", "RTP_LLM_ENABLE_SCR", "SCR_ENABLE")
    )
    return enabled and os.environ.get("SCR_PHASE", "").strip().lower() in {
        "checkpoint",
        "restore",
    }


class ReportWorker(object):
    def __init__(self, *args):
        super(ReportWorker, self).__init__(*args)
        self._runtime_tags = HippoHelper.get_hippo_tags()
        self._configured_tags = self.parse_kmon_tags(os.environ.get("kmonitorTags", ""))
        self.init_tags = self._runtime_tags.copy()
        self.init_tags.update(self._configured_tags)
        logging.info(
            f"kmonitor report default tags: {json.dumps(self.init_tags, indent=4)}"
        )
        self.metrics: Dict[str, MetricBase] = {}
        self.metric_lock: Lock = Lock()
        self.started = False
        self._report_thread: Thread | None = None
        self._transport_deferred = False
        if HippoHelper.is_hippo_env():
            self.flume = None
            self._transport_deferred = _defer_transport_for_scr()
            if self._transport_deferred:
                logging.info("defer kmonitor transport until SCR steady-point returns")
            else:
                self._activate_hippo_transport(refresh_identity=False)
        else:
            self.flume = None
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
        report_tags = data_point.tags.copy()
        # Runtime identity is authoritative after restore. Metric instances can
        # predate the restore and therefore still contain seed identity tags.
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

    def _activate_hippo_transport(self, *, refresh_identity: bool = True) -> None:
        if refresh_identity:
            self._runtime_tags = HippoHelper.refresh_runtime_identity()
            # KMonitor instances retain this dict by reference. Update it in
            # place so metrics registered after restore also use fresh tags.
            self.init_tags.clear()
            self.init_tags.update(self._runtime_tags)
            self.init_tags.update(self._configured_tags)
        report_host = os.environ.get("HIPPO_SLAVE_IP", "localhost")
        if self.flume is not None:
            self.flume.close()
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

    def pause_for_checkpoint(self) -> bool:
        was_started = self.started
        self.stop()
        if self._report_thread is not None:
            self._report_thread.join(timeout=5)
            if self._report_thread.is_alive():
                self.started = was_started
                raise RuntimeError("Kmonitor reporter did not quiesce before SCR")
        try:
            if self.flume is not None:
                self.flume.close()
        except Exception:
            if was_started:
                self.start()
            raise
        return was_started

    def resume_after_checkpoint(self, was_started: bool) -> None:
        if HippoHelper.is_hippo_env() and (was_started or self._transport_deferred):
            self._activate_hippo_transport()
        elif was_started:
            try:
                if self.flume is not None:
                    self.flume.reconnect()
            finally:
                self.start()


report_worker = ReportWorker()
