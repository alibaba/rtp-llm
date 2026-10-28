"""Native Master configuration binding and per-Master state reads."""
import copy
import json

from .records import write_json
from .transport import ExecutionError, post


def master_environment(config, topology, directory):
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
            if not addresses:
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
    write_json(directory / 'native-environment.json', env)
    return env


async def invoke_master(urls, endpoint, payload):
    """Read each native Master separately; all other calls keep their normal binding."""
    interface = payload['interface']
    if interface not in ('master/leadership/get', 'master/engine/get'):
        return await post(endpoint, payload)
    masters = [url for node, url in urls.items() if node.startswith('master-')]
    if not masters:
        raise ExecutionError('No running Master endpoint')
    instances = [] if interface == 'master/leadership/get' else {}
    for url in masters:
        value = (await post(url, payload))['instances']
        if interface == 'master/leadership/get':
            if not isinstance(value, list):
                raise ExecutionError('Master leadership must return a list')
            instances.extend(value)
        else:
            if not isinstance(value, dict):
                raise ExecutionError('Master engines must return a per-Master mapping')
            if set(instances) & set(value):
                raise ExecutionError('Duplicate Master identity in read results')
            instances.update(value)
    return {'instances': instances}
