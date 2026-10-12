"""Compile author-owned cases and explain their effective contracts, without I/O."""

import copy


def validate_cases(source, *, profile=None):
    from scenario import load_scenarios, compile_scenarios
    from scenario.catalog import handlers
    from case_api.registry import registry
    from scenario.config_data import merge_data
    from monitoring.query_plan import definitions, SOURCE_KINDS

    plans = compile_scenarios(load_scenarios(source), profile, handlers())
    if not plans:
        raise ValueError('no matching case instances')
    instances = []
    for plan in plans:
        implementation = plan['implementation']
        configuration = implementation['configuration']
        parameters = copy.deepcopy(configuration['parameters'])
        for variant in configuration.get('variants', []):
            if variant['id'] == plan['variant_id']:
                parameters = merge_data(parameters, variant.get('parameters', {}))
        capability = registry().cases[implementation['program']].definition
        frozen = implementation.get('monitoring_query_plan')
        metric_plan = frozen['definition'] if frozen else None
        instances.append(dict(
            id=plan['id'], source=plan['source_path'],
            parameters=parameters,
            numeric_parameters=implementation['numeric_parameters'][plan['variant_id']],
            stages=[dict(id=stage['id'], action=stage['action'],
                         timeout_s=stage['timeout_s'], check_ids=stage['check_ids'])
                    for stage in plan['stages']],
            windows=parameters.get('observation', {}).get('windows', {}),
            metrics=dict(definitions=definitions(metric_plan), consumers=dict(metric_plan.get('demand', {}),
                         collector_health=sorted(kind + '/up' for kind in SOURCE_KINDS)))
                    if metric_plan else dict(definitions={}, consumers={}),
            reports=plan.get('reports', ['default.yaml']),
            reporting=plan.get('reporting'),
            execution=plan['execution'], resource_budget=plan['resource_budget'],
            capabilities=dict(actions=[action.name for action in capability.actions],
                              producers=list(capability.producers), sources=list(capability.sources),
                              report_views=list(capability.report_views)),
        ))
    return dict(case_validation_schema_version=1, status='PASS',
                scope='static_compilation', runtime_validation='NOT_RUN', instances=instances)
