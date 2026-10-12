"""分层 sidecar 的存储契约；读取边界将历史格式投影为统一运行字段。"""
import copy


def normalize(document):
    if not isinstance(document, dict):
        raise ValueError('traffic manifest must be an object')
    if 'traffic_manifest_schema_version' not in document:
        if document.get('manifest_schema_version') != 2:
            raise ValueError('unsupported traffic manifest version')
        return copy.deepcopy(document)
    if document.get('traffic_manifest_schema_version') != 1 or set(document) != {
            'traffic_manifest_schema_version', 'artifact', 'capture', 'transformation', 'statistics'}:
        raise ValueError('invalid layered traffic manifest')
    artifact = document['artifact']
    capture = document['capture']
    transform = document['transformation']
    if (not isinstance(artifact, dict) or set(artifact) != {'bytes', 'sha256', 'codec', 'metadata'}
            or not isinstance(capture, dict) or set(capture) != {'source', 'execution', 'requested_window', 'observed_window', 'provenance', 'limitations'}
            or not isinstance(transform, dict) or set(transform) != {'status', 'tool', 'inputs'}
            or transform['status'] not in {'recorded', 'legacy_unverified'}
            or not isinstance(transform['inputs'], list)):
        raise ValueError('invalid traffic manifest sections')
    execution = capture['execution']
    if (not isinstance(execution, dict) or set(execution) != {'status', 'started_at', 'finished_at', 'tool', 'issues', 'raw_sources'}
            or execution['status'] not in {'recorded', 'unavailable'}
            or not isinstance(execution['issues'], list) or not isinstance(execution['raw_sources'], list)):
        raise ValueError('invalid capture execution facts')
    if execution['status'] == 'recorded':
        from datetime import datetime
        start, end = (datetime.fromisoformat(execution[k]) for k in ('started_at', 'finished_at'))
        if start.tzinfo is None or end.tzinfo is None or start > end or not execution['tool']:
            raise ValueError('capture execution requires timezone-aware interval and tool')
    elif any(execution[k] is not None for k in ('started_at', 'finished_at', 'tool')):
        raise ValueError('unavailable execution cannot invent capture timestamps')
    if transform['status'] == 'recorded' and (not transform['tool'] or not transform['inputs']):
        raise ValueError('recorded transformation requires tool and input evidence')
    if not isinstance(artifact['metadata'], dict) or 'version' in artifact['metadata'] or 'provenance' in artifact['metadata']:
        raise ValueError('codec version and capture provenance have separate owners')
    codec = artifact['codec']
    if not isinstance(codec, dict) or set(codec) != {'name', 'version'}:
        raise ValueError('invalid traffic codec')
    result = copy.deepcopy(artifact['metadata'])
    result.update(manifest_schema_version=2, version=codec['version'], codec=copy.deepcopy(codec),
                  bytes=artifact['bytes'], sha256=artifact['sha256'], data_kind='real',
                  source=copy.deepcopy(capture['source']), provenance=copy.deepcopy(capture['provenance']),
                  capture_window=copy.deepcopy(capture['observed_window']),
                  limitations=copy.deepcopy(capture['limitations']), statistics=copy.deepcopy(document['statistics']))
    p = result['provenance']
    if not isinstance(p, dict) or not isinstance(document['statistics'], dict):
        raise ValueError('invalid capture provenance or statistics')
    from datetime import datetime, timedelta, timezone
    zone = timezone(timedelta(hours=8))
    observed = {key: datetime.fromtimestamp(p[field] / 1000, zone).isoformat()
                if p.get(field) is not None else None
                for key, field in (('start', 'source_start'), ('end', 'source_end'))}
    if capture['observed_window'] != dict(observed, timezone='Asia/Shanghai'):
        raise ValueError('observed window conflicts with embedded provenance')
    expected = {key: p.get(field) for key, field in (('start_epoch_ms', 'capture_start'), ('end_epoch_ms', 'capture_end'))}
    if capture['requested_window'] != expected:
        raise ValueError('requested capture window conflicts with embedded provenance')
    return result


def layered(manifest, *, previous=None, transformation=None, capture_execution=None):
    """Rebuild artifact/statistics while retaining independently supplied facts."""
    metadata = {key: copy.deepcopy(value) for key, value in manifest.items()
                if key not in {'version', 'provenance', 'manifest_schema_version', 'bytes', 'sha256',
                               'codec', 'data_kind', 'source', 'capture_window', 'statistics', 'limitations'}}
    p = copy.deepcopy(manifest['provenance'])
    document = dict(traffic_manifest_schema_version=1,
        artifact=dict(bytes=manifest['bytes'], sha256=manifest['sha256'], codec=copy.deepcopy(manifest['codec']), metadata=metadata),
        capture=dict(source=copy.deepcopy(manifest['source']), provenance=p,
            execution=copy.deepcopy(capture_execution or dict(status='unavailable', started_at=None, finished_at=None, tool=None, issues=['capture execution history unavailable'], raw_sources=[])),
            requested_window=dict(start_epoch_ms=p.get('capture_start'), end_epoch_ms=p.get('capture_end')),
            observed_window=copy.deepcopy(manifest['capture_window']), limitations=copy.deepcopy(manifest['limitations'])),
        transformation=copy.deepcopy(transformation or dict(status='legacy_unverified', tool=None, inputs=[])),
        statistics=copy.deepcopy(manifest['statistics']))
    if previous:
        old = normalize(previous)
        if old['sha256'] != manifest['sha256']:
            raise ValueError('cannot retain capture facts for changed traffic bytes')
        document['capture']['limitations'] = copy.deepcopy(old.get('limitations', []))
        if 'traffic_manifest_schema_version' in previous:
            if transformation is None:
                document['transformation'] = copy.deepcopy(previous['transformation'])
            if capture_execution is None:
                document['capture']['execution'] = copy.deepcopy(previous['capture']['execution'])
    normalize(document)
    return document
