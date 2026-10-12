"""Shared options for CI planning and execution; no selection defaults here."""
from scenario.suites import SuiteCatalog


def add_ci_selection(parser):
    catalog = SuiteCatalog.read()
    parser.add_argument('--job', required=True, choices=tuple(catalog.ci_jobs))
    parser.add_argument('--suite', choices=tuple(catalog.suites), help='override the declared suite')
    parser.add_argument('--profiles', help='comma-separated profiles; defaults to the job policy')
    parser.add_argument('--instances', help='exact IDs override suite/profile selection within supported job profiles')
    parser.add_argument('--worker', help='select one configured worker')
    parser.add_argument('--grade', choices=('strict', 'normal', 'loose'), default='normal')


def resolve_ci_selection(args):
    from scenario.ci_plan import CiPlanner
    profiles = None if args.profiles is None else [value.strip() for value in args.profiles.split(',')]
    plan = CiPlanner().build(args.job, suite=args.suite, profiles=profiles,
                             instances=args.instances, grade=args.grade)
    if not plan.identities(args.worker):
        from scenario.loader import ScenarioError
        raise ScenarioError('selected CI worker has no instances')
    return plan
