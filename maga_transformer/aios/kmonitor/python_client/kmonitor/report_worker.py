
from threading import Thread, Lock
from multiprocessing import Process
from typing import Dict, List

import time
import json
import os
import logging
import traceback

from maga_transformer.aios.kmonitor.python_client.flume.pyflume import FlumeClient
from maga_transformer.aios.kmonitor.python_client.flume.ttypes import ThriftFlumeEvent
from maga_transformer.aios.kmonitor.python_client.kmonitor.utils.hippo_helper import HippoHelper
from maga_transformer.aios.kmonitor.python_client.kmonitor.metrics.metric_base import MetricBase, MetricDataPoint

_ReportWorker__REPORT_HOST = os.getenv("HIPPO_SLAVE_IP", 'localhost')
_ReportWorker__REPORT_PORT = 4141
_ReportWorker__FLUME_CLIENT_TIMEOUT_MS = 1000

_ReportWorker__REPORT_INTERVAL_SECOND = 1
_ReportWorker__REPORT_HEADERS = {
    'topic': 'py_kmonitor'
}
_ReportWorker__REPORT_KMONITOR_MULTI_SEP = '@'
_ReportWorker__REPORT_KMONITOR_KEYVALUE_SEP = '^'


class ReportWorker(object):
    def __init__(self, *args):
        super(ReportWorker, self).__init__(*args)
        self.init_tags = HippoHelper.get_hippo_tags()
        self.init_tags.update(self.parse_kmon_tags(os.environ.get('kmonitorTags', '')))
        logging.info(f"kmonitor report default tags: {json.dumps(self.init_tags, indent=4)}")
        self.metrics: Dict[str, MetricBase] = {}
        self.metric_lock: Lock = Lock()
        self.started = False
        if HippoHelper.is_hippo_env():
            self.flume = FlumeClient(_ReportWorker__REPORT_HOST, _ReportWorker__REPORT_PORT,
                                     timeout=_ReportWorker__FLUME_CLIENT_TIMEOUT_MS)
            self.start()
            logging.info(f'hippo role [{HippoHelper.role}] at host [{HippoHelper.host_ip}-{HippoHelper.container_ip}] '
                          'started reporting kmonitor.')
        else:
            self.flume = None
            self.start()
            logging.info('test mode, kmonitor metrics not reported.')

    def parse_kmon_tags(self, kmon_tags_str: str) -> Dict[str, str]:
        kmon_tags: Dict[str, str] = {}
        if not kmon_tags_str:
            return {}
        for tag in kmon_tags_str.split(_ReportWorker__REPORT_KMONITOR_MULTI_SEP):
            kv = tag.split(_ReportWorker__REPORT_KMONITOR_KEYVALUE_SEP)
            if len(kv) != 2:
                logging.error(f'kmon parse tags failed: tag can not split: {tag}')
                return {}
            kmon_tags[kv[0].strip()] = kv[1].strip()
        return kmon_tags

    def register_metric(self, metric: MetricBase) -> None:
        with self.metric_lock:
            if metric.name in self.metrics.keys():
                raise Exception(f'metric {metric.name} already registered, can not register again.')
            self.metrics[metric.name] = metric

    def render_event(self, metric_name: str, timestamp: int, data_point: MetricDataPoint) -> ThriftFlumeEvent:
        value_str = str(data_point.value)
        tag_str : str = ' '.join(['='.join([k, v]) for (k, v) in list(data_point.tags.items())])
        report_message : bytes = ' '.join([metric_name, str(timestamp), value_str, tag_str]).encode('utf-8')
        return ThriftFlumeEvent(_ReportWorker__REPORT_HEADERS, report_message)

    def get_report_events(self) -> List[ThriftFlumeEvent]:
        events: List[ThriftFlumeEvent] = []
        timestamp : int = int(round(time.time()))
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
            logging.error(f'kmonitor report thread error: {e} {traceback.format_exc()}')
        logging.warn('kmonitor report process exited.')

    def start(self) -> None:
        self.started = True
        report_thread = Thread(target=self.report_cycle)
        report_thread.daemon = True
        report_thread.start()

    def stop(self) -> None:
        self.started = False

report_worker = ReportWorker()
