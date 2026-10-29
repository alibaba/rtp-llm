"""Validate local launch recipes independently of process ownership."""
from pathlib import Path
from ..config.values import duration, fields, text
from .transport import ExecutionError
from .monitoring import prepare_prometheus
from .engines import ROLES


def prepare_launch(raw, case, base):
    """Launch recipes are developer bindings, separate from case business configuration."""
    fields(raw, {'components', 'prometheus', 'zookeeper', 'engine_mode'}, {'components'}, where='launch')
    if not isinstance(raw['components'], dict):
        raise ExecutionError('launch components must be a mapping')
    if 'zookeeper' in raw['components']:
        raise ExecutionError('ZooKeeper belongs in the launch.zookeeper recipe, not components')
    components = {}
    recipes = dict(raw['components'])
    if 'zookeeper' in raw:
        recipes['zookeeper'] = raw['zookeeper']
    for role, recipe in recipes.items():
        if role != 'zookeeper' and role not in case.environment:
            raise ExecutionError(f'Launch recipe names an unconfigured component: {role}')
        fields(recipe, {'command', 'cwd', 'env', 'startup_timeout'}, {'command'}, where='launch.' + role)
        command = recipe['command']
        if not isinstance(command, list) or not command or any(not isinstance(v, str) or not v for v in command):
            raise ExecutionError('command must be a nonempty argv list; no shell expansion')
        env = recipe.get('env', {})
        if not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()):
            raise ExecutionError('Launch env must contain string keys and values')
        if any(k.startswith('MOCK_') for k in env):
            raise ExecutionError('MOCK_* launch variables are assigned by the runner')
        startup = recipe.get('startup_timeout', '30s')
        duration(startup, positive=True)
        cwd = Path(base) / text(recipe.get('cwd', '.'), 'launch.cwd')
        if not cwd.is_dir():
            raise ExecutionError(f'Launch cwd is not a directory: {cwd}')
        deployment = case.environment[role]['deployment'] if role != 'zookeeper' else {}
        if deployment.get('resources') or deployment.get('heap'):
            raise ExecutionError('This local launcher does not yet enforce resources/heap; refusing to ignore them')
        components[role] = {**recipe, 'command': command, 'env': env, 'cwd': str(cwd.resolve()),
                            'startup_timeout': startup}
    zookeeper = components.pop('zookeeper', None)
    native_master = 'MODEL_SERVICE_CONFIG' in case.environment.get('master', {}).get('config', {})
    if native_master != (zookeeper is not None):
        raise ExecutionError('Native Master requires a zookeeper launch recipe; other components do not use it')
    missing = set(case.environment) - set(components)
    if missing:
        raise ExecutionError(f'Missing component launch recipes: {sorted(missing)}')
    mode = raw.get('engine_mode', 'shared')
    if mode not in ('shared', 'isolated'):
        raise ExecutionError('engine_mode must be shared or isolated')
    engine_recipes = [recipe for role, recipe in components.items() if role in ROLES]
    if mode == 'shared' and engine_recipes and any(recipe != engine_recipes[0] for recipe in engine_recipes[1:]):
        raise ExecutionError('Shared Engine JVM requires identical command, cwd, env and startup_timeout for all Engine roles; put instance settings in config')
    result = {'components': components, 'engine_mode': mode}
    if zookeeper is not None:
        result['zookeeper'] = zookeeper
    if 'prometheus' in raw:
        result['prometheus'] = prepare_prometheus(raw['prometheus'], base)
        if any('prometheus' in value.get('config', {}) for value in case.environment.values()):
            raise ExecutionError('Managed Prometheus cannot be combined with component prometheus configuration')
    return result
