"""Pure report section and metadata construction, without publication I/O."""

def details(title, value, *, opened=False, identity=None):
    return dict(type="details", id=identity, title=title, value=value, opened=opened)


def table(title, columns, rows, *, opened=True, identity=None):
    return dict(type="table", id=identity, title=title, columns=list(columns), rows=list(rows), opened=opened)


def links(title, items):
    return dict(type="links", title=title, items=list(items))


def run_meta(
    identity,
    *,
    implementation=None,
    workload=None,
    configuration=None,
    environment=None,
    clock=None,
    evidence=None,
    runs=None,
):
    """None means unknown, never proof of matching experiment conditions."""
    metadata = dict(
        run_meta_schema_version=1,
        identity=identity,
        implementation=implementation,
        workload=workload,
        configuration=configuration,
        environment=environment,
        clock=clock,
        evidence=evidence,
    )
    if runs is not None:
        metadata["runs"] = runs
    return metadata

