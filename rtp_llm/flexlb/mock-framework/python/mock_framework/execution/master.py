"""Native Master configuration binding and per-Master state reads."""
import copy
import json

from .records import write_json
from .transport import ExecutionError


def _environment(config, topology, directory, *, require_engines):
    env = {}
    for key, value in config.items():
        if key.isupper():
            if not isinstance(value, str):
                raise ExecutionError(f'Master native setting {key} must be an environment string')
            env[key] = value
    if 'MODEL_SERVICE_CONFIG' not in env:
        return env  # Generic transport probes do not launch the native Master.
    try:
        route = json.loads(env['MODEL_SERVICE_CONFIG'])
        groups = route['role_endpoints']
        if not isinstance(groups, list) or len(groups) != 1:
            raise ValueError('local Master currently requires exactly one endpoint group')
        route = copy.deepcopy(route)
        hosts = {}
        configured = {node['role'] for node in topology}
        for role in ('prefill', 'decode', 'vit'):
            endpoint = route['role_endpoints'][0].get(role + '_endpoint')
            if endpoint is None:
                if role in configured:
                    raise ValueError(f'Missing native {role}_endpoint')
                continue
            if endpoint.get('protocol') != 'grpc' or not isinstance(endpoint.get('address'), str) or not endpoint['address']:
                raise ValueError('local Engine endpoints require grpc protocol and a discovery address')
            domain = endpoint['address']
            if domain in hosts:
                raise ValueError('Each Engine role needs a distinct discovery address')
            addresses = []
            for node in topology:
                if node['role'] == role:
                    host, port = node['rpc'].rsplit(':', 1)
                    # Native WorkerAddressService translates grpc discovery ports to its HTTP identity.
                    port = int(port)
                    if port <= 0:
                        raise ValueError('Invalid Engine RPC port')
                    addresses.append(f'{host}:{port}')
            if require_engines and not addresses:
                raise ValueError(f'No running Engines for native {role}_endpoint')
            hosts[domain] = addresses
        if not hosts or route['role_endpoints'][0].get('pd_fusion_endpoint') is not None:
            raise ValueError('local Master supports separate P/D/V endpoints only')
        discovery = directory / 'discovery.json'
        write_json(discovery, hosts)
        route.pop('hosts', None)
        route['discovery_file'] = str(discovery)
        env['MODEL_SERVICE_CONFIG'] = json.dumps(route, ensure_ascii=False)
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise ExecutionError(f'Cannot bind local MODEL_SERVICE_CONFIG: {error}') from error
    return env


def master_environment(config, topology, directory, *, require_engines=True):
    env = _environment(config, topology, directory, require_engines=require_engines)
    write_json(directory / 'native-environment.json', env)
    return env


def refresh_discovery(config, topology, directory):
    if 'MODEL_SERVICE_CONFIG' in config:
        # The consumer owns reload timing; only publish the current addresses here.
        _environment(config, topology, directory, require_engines=False)
