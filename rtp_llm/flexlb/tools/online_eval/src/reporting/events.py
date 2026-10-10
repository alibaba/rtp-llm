"""Project recorded event times; view definitions only select and label them."""

import copy
import math
import re


def validate_events(path, charts, fail):
    declarations = charts.get('events', {})
    if not isinstance(declarations, dict):
        fail(str(path) + '.charts.events', 'expected event mapping')
    for identity, declaration in declarations.items():
        location = str(path) + '.charts.events.' + str(identity)
        if (type(identity) is not str or not re.fullmatch(r'[a-z][a-z0-9_]*', identity)
                or not isinstance(declaration, dict)
                or type(declaration.get('label')) is not str or not declaration['label'].strip()):
            fail(location, 'invalid event identity or label')
        source = declaration.get('source')
        required = {'label', 'source', 'stage', 'boundary'} if source == 'stage' else {'label', 'source', 'event'}
        if source not in {'stage', 'case'} or declaration.keys() != required:
            fail(location, 'event requires an explicit stage or case source')
        selector = declaration['stage'] if source == 'stage' else declaration['event']
        if type(selector) is not str or not re.fullmatch(r'[a-z][a-z0-9_]*', selector):
            fail(location, 'invalid event source identity')
        if source == 'stage' and declaration['boundary'] not in {'start', 'end'}:
            fail(location, 'stage boundary must be start or end')
    selections = list(charts.get('panels', []))
    if 'event_ids' in charts:
        selections.append(dict(event_ids=charts['event_ids']))
    for panel in selections:
        selected = panel.get('event_ids', [])
        if (not isinstance(selected, list) or any(type(identity) is not str or identity not in declarations
                                                for identity in selected)
                or len(set(selected)) != len(selected)):
            fail(str(path) + '.charts.panels', 'event_ids must select declared events without duplicates')


def project_events(presentation, *, origin, phases=(), events=()):
    if type(origin) not in (int, float) or not math.isfinite(origin):
        raise ValueError('event origin must be finite epoch seconds')
    projected = []
    for identity, declaration in presentation['charts'].get('events', {}).items():
        source = declaration['source']
        rows = (row for row in phases if row['stage'] == declaration['stage']
                and row['event'] == declaration['boundary']) if source == 'stage' else (
                    row for row in events if row['id'] == declaration['event'])
        for row in rows:
            epoch = row.get('epoch_s')
            if type(epoch) not in (int, float) or not math.isfinite(epoch):
                raise ValueError('recorded event lacks finite epoch_s: ' + identity)
            label = declaration['label']
            if source == 'stage' and row.get('status') in {'FAIL', 'ERROR', 'TIMEOUT', 'BLOCKED'}:
                label += ' · ' + row['status']
            projected.append(dict(id=identity, name=label, t=epoch-origin,
                                  epoch_s=epoch, source=dict(type=source, record=copy.deepcopy(row))))
    return sorted(projected, key=lambda event: (event['t'], event['id']))


def attach_events(spec, presentation, *, origin, phases=(), events=()):
    """Freeze all selected markers, then explicitly select markers for each panel."""
    markers = project_events(presentation, origin=origin, phases=phases, events=events)
    spec['timeOriginEpochS'] = origin
    spec['events'] = markers
    descriptors = {panel['id']: panel for panel in presentation['charts'].get('panels', [])}
    for panel in spec['panels']:
        ids = descriptors.get(panel['id'], {}).get('event_ids', presentation['charts'].get('event_ids', []))
        panel['events'] = [copy.deepcopy(event) for event in markers if event['id'] in ids]
    return spec


def validate_stage_sources(presentation, stage_ids, path):
    """A missing runtime occurrence is allowed; an unknown configured stage is not."""
    from scenario.validation import fail
    for identity, declaration in presentation['charts'].get('events', {}).items():
        if declaration['source'] == 'stage' and declaration['stage'] not in stage_ids:
            fail(path, 'event ' + identity + ' references unknown stage ' + declaration['stage'])
