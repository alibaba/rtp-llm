"""Table-driven descriptions retain case ownership of measurements and accuracy."""
from dataclasses import dataclass
from types import MappingProxyType
from copy import deepcopy


@dataclass(frozen=True)
class MeasurementDescription:
    function: object
    source: str
    population: str
    accuracy: str
    request_identity: bool
    collection: object = None

    def contract(self):
        from monitoring.contracts import implementation_measurement
        from implementation_identity import implementation_identity
        result = dict(source_type=self.source, implementation=implementation_identity(self.function),
            measurement=implementation_measurement(self.function, population=self.population,
                accuracy=self.accuracy, request_identity=self.request_identity))
        if self.collection is not None:
            result["collection"] = deepcopy(self.collection)
        return result


class ProducerDescriptions:
    def __init__(self, declarations):
        if any(not isinstance(key, tuple) or len(key) != 2 or
               not isinstance(value, MeasurementDescription) for key, value in declarations.items()):
            raise ValueError("descriptions require (producer, metric) keys and MeasurementDescription values")
        self.declarations = MappingProxyType(dict(declarations))

    def describe(self, producer, identity, calculation):
        if calculation is not None:
            raise ValueError("producer owns its case-specific calculation")
        key = (producer, identity)
        if key not in self.declarations:
            raise ValueError("unknown producer metric: " + producer + "/" + identity)
        return self.declarations[key].contract()
