"""Single owner of instance addresses and lifetime history. Readers receive detached snapshots."""
import copy
from .records import timestamp, write_json
from .transport import ExecutionError


class Topology:
    def __init__(self, directory):
        self.directory = directory
        self._active, self._history, self._used = {}, [], set()
        self._publishers = []

    def active(self, role=None):
        return copy.deepcopy([node for node in self._active.values() if role is None or node['role'] == role])

    def history(self, role=None):
        return copy.deepcopy([node for node in self._history if role is None or node['role'] == role])

    def add_publisher(self, publish):
        self._publishers.append(publish)

    def publish(self):
        snapshot = self.active()
        write_json(self.directory / 'topology.json', snapshot)
        write_json(self.directory / 'topology-history.json', self.history())
        for publish in self._publishers:
            publish(copy.deepcopy(snapshot))

    def register(self, name, role, control, endpoint=None):
        if name in self._used:
            raise ExecutionError(f'Instance identity has already been used: {name}')
        if endpoint is not None:
            if endpoint.get('id', name) != name or endpoint.get('role', role) != role or endpoint.get('control') != control:
                raise ExecutionError('Instance endpoints differ from assigned identity')
            endpoint = {**copy.deepcopy(endpoint), 'id': name, 'role': role}
            if not isinstance(endpoint.get('rpc'), str) or not endpoint['rpc'] or any(node['rpc'] == endpoint['rpc'] for node in self._history):
                raise ExecutionError('Instance RPC address is missing or has already been used')
            self._active[name] = endpoint
            self._history.append({**copy.deepcopy(endpoint), 'started_at': timestamp()})
        self._used.add(name)
        self.publish()

    def retire(self, name):
        if name not in self._active:
            raise ExecutionError(f'Unknown active instance: {name}')
        self._active.pop(name, None)
        for node in self._history:
            if node['id'] == name:
                node['stopped_at'] = timestamp()
        self.publish()

    def clear(self):
        self._active.clear()
        for node in self._history:
            node.setdefault('stopped_at', timestamp())
        self.publish()
