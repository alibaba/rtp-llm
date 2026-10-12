"""性能采集包：配置证据、测量证据与 Mock 参数分离，不访问在线平台。"""
import hashlib
import json
from pathlib import Path

ROLES = {'prefill', 'decode', 'master', 'frontend'}
LAYERS = {'template', 'biz', 'deployment', 'carbon_target', 'process_readback'}


def whale_read_plan(deployment, images):
    """Use the deployment's pinned versions; image candidates are not role commits."""
    if (deployment.get('id') != images.get('deployment_id')
            or deployment.get('py_biz_id') != images.get('biz_id')
            or deployment.get('biz_version') != images.get('biz_version')):
        raise ValueError('Whale deployment and image version evidence disagree')
    if not images.get('template_version_id'):
        raise ValueError('deployed template version is unknown')
    zones = deployment.get('zone_deployments')
    if not isinstance(zones, list) or any(not z.get('zone_name') for z in zones):
        raise ValueError('Whale deployment lacks zone identities')
    return [
        dict(operation='getPyBizTemplateVersionById', parameters=dict(id=images['template_version_id'])),
        dict(operation='getPyBizDistributeVersion', parameters=dict(id=images['biz_id'], biz_version=str(images['biz_version']))),
        dict(operation='getPyBizDeploymentById', parameters=dict(id=images['deployment_id'])),
        *[dict(operation='getZoneRealCarbonTarget', parameters=dict(id=images['deployment_id'], zone_name=z['zone_name'])) for z in zones],
    ]


def validate(document, root):
    """Validate explicitly supplied evidence. Missing history remains unavailable."""
    if not isinstance(document, dict) or set(document) != {
            'performance_capture_schema_version', 'identity', 'collection', 'configuration', 'measurements', 'derived'}:
        raise ValueError('invalid performance capture fields')
    if document['performance_capture_schema_version'] != 1:
        raise ValueError('unsupported performance capture version')
    identity = document['identity']
    if not isinstance(identity, dict) or set(identity) != {'id', 'model', 'hardware', 'deployment_ref'} or any(
            type(identity[k]) is not str or not identity[k] for k in identity):
        raise ValueError('performance identity requires opaque deployment_ref, model and hardware')
    collection = document['collection']
    if not isinstance(collection, dict) or set(collection) != {'status', 'started_at', 'finished_at', 'tool', 'issues'}:
        raise ValueError('invalid performance collection')
    if collection['status'] not in {'complete', 'partial', 'legacy_unverified'} or not isinstance(collection['issues'], list):
        raise ValueError('invalid performance collection status')
    if collection['status'] != 'legacy_unverified':
        from datetime import datetime
        start, end = (datetime.fromisoformat(collection[k]) for k in ('started_at', 'finished_at'))
        if start.tzinfo is None or end.tzinfo is None or start > end or not collection['tool']:
            raise ValueError('collection requires timezone-aware interval and tool')
    root = Path(root).resolve()
    def reference(ref):
        if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'} or type(ref['path']) is not str:
            raise ValueError('evidence requires path and sha256')
        path = (root / ref['path']).resolve()
        if Path(ref['path']).is_absolute() or not path.is_relative_to(root) or not path.is_file():
            raise ValueError('capture evidence missing or escapes bundle')
        hasher = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if ref['sha256'] != digest:
            raise ValueError('capture evidence checksum mismatch')
    configuration = document['configuration']
    if not isinstance(configuration, dict) or set(configuration) != LAYERS:
        raise ValueError('configuration requires all layer slots, with explicit availability')
    for layer, record in configuration.items():
        if not isinstance(record, dict) or set(record) != {'status', 'reason', 'evidence', 'roles'}:
            raise ValueError('invalid configuration layer')
        if record['status'] not in {'captured', 'unavailable', 'not_applicable'}:
            raise ValueError('invalid layer status')
        if record['status'] == 'captured':
            reference(record['evidence'])
        elif record['evidence'] is not None or not record['reason']:
            raise ValueError('missing layer requires reason and no invented evidence')
        if not isinstance(record['roles'], dict) or set(record['roles']) != ROLES:
            raise ValueError('configuration must declare all role slots')
        for role in record['roles'].values():
            if not isinstance(role, dict) or set(role) != {'status', 'image', 'commit', 'version_basis'}:
                raise ValueError('invalid role version evidence')
            if role['status'] not in {'recorded', 'unavailable', 'not_applicable'}:
                raise ValueError('invalid role status')
            if role['status'] == 'recorded':
                if record['status'] != 'captured' or not role['image'] or role['version_basis'] not in {'declared', 'carbon_target', 'process_readback'}:
                    raise ValueError('role versions require image and explicit evidence basis')
                if role['version_basis'] != {'carbon_target': 'carbon_target', 'process_readback': 'process_readback'}.get(layer, 'declared'):
                    raise ValueError('declared config cannot claim runtime verification')
            elif any(role[k] is not None for k in ('image', 'commit', 'version_basis')):
                raise ValueError('unavailable role must not invent version fields')
    for group in ('measurements', 'derived'):
        if not isinstance(document[group], dict):
            raise ValueError('invalid performance evidence group')
        for row in document[group].values():
            required = {'evidence', 'configuration_layers', 'method', 'unit'} if group == 'measurements' else {'evidence', 'measurement_ids', 'method'}
            if not isinstance(row, dict) or set(row) != required or not row['method']:
                raise ValueError('invalid measurement or derived record')
            reference(row['evidence'])
            if group == 'measurements':
                if not row['unit'] or not isinstance(row['configuration_layers'], list) or not row['configuration_layers'] or any(
                        k not in configuration or configuration[k]['status'] != 'captured' for k in row['configuration_layers']):
                    raise ValueError('measurement requires captured configuration references and unit')
            elif not isinstance(row['measurement_ids'], list) or not row['measurement_ids'] or any(k not in document['measurements'] for k in row['measurement_ids']):
                raise ValueError('derived model requires declared measurements')
    return document


def write_manifest(path, document):
    """Publish the index last; supplied evidence must already exist in the bundle."""
    path = Path(path)
    validate(document, path.parent)
    if path.exists():
        raise ValueError('refusing to overwrite performance capture manifest')
    with path.open('x', encoding='utf-8') as target:
        json.dump(document, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write('\n')


def read_manifest(path):
    path = Path(path)
    return validate(json.loads(path.read_text()), path.parent)
