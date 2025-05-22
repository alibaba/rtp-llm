from abc import abstractmethod
from typing import Dict, List

class MetricDataPoint(object):
    def __init__(self, value: float, tags: Dict[str, str]):
        self.value = value
        self.tags = tags


class MetricBase(object):
    def __init__(self, name: str, tags: Dict[str, str]):
        super(MetricBase, self).__init__()
        self.name: str = name
        self.tags: Dict[str, str] = tags

    @abstractmethod
    def report(self, value: float = 1, tags: Dict[str, str] = {}) -> None:
        pass

    @abstractmethod
    def fetch_reported_data(self) -> List[MetricDataPoint]:
        pass
