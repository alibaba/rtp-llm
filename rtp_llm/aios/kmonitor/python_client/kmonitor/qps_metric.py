
class QpsMetric:
    def __init__(self, metric_name: str, metric_tags: dict) -> None:
        self.last_record_time = time.time()
        self.query_count = 0
        self.metric_name = metric_name
        self.metric_tags = metric_tags
        self.lock = Lock()

    def try_get_qps_to_report(self) -> Union[float, None]:
        self.lock.acquire()
        current_time = time.time()
        if (time.time() - self.last_record_time) >= QPS_REPORT_INTERVAL_SECOND:
            logging.debug(f'recorded {self.query_count} qps '
                         f'in {current_time - self.last_record_time:.4f} seconds ')
            qps = self.query_count / (current_time - self.last_record_time)
            self.query_count = 0
            self.last_record_time = current_time
            self.lock.release()
            return qps
        else:
            self.lock.release()
            return None

    def record_qps(self) -> None:
        self.lock.acquire()
        self.query_count += 1
        self.lock.release()
