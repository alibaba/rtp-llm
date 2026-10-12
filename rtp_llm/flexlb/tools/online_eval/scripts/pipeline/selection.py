"""Shared CLI selection defaults for listing and execution."""
from scenario.suites import SuiteCatalog
from scenario.execution_policy import KINDS
from flexlb_cfg import PROFILES


def add_selection_arguments(parser, *, allow_all_profiles=True):
    catalog = SuiteCatalog.read()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--suite', choices=tuple(catalog.suites), help='explicit case::variant suite')
    group.add_argument('--kind', choices=KINDS, help='all variants of this kind')
    group.add_argument('--all', action='store_true', help='all cases and variants')
    parser.add_argument('--profile', choices=(*PROFILES, 'all') if allow_all_profiles else PROFILES, help='profile; all expands every supported profile')
    parser.add_argument('--master-mode', choices=('sb', 'sn', 'wb', 'wn'))


def resolve_selection(args, parser):
    catalog = SuiteCatalog.read()
    if args.suite is None and args.kind is None and not args.all:
        args.suite = catalog.default_suite
    if args.master_mode:
        from mode_profiles import resolve_mode
        profile = resolve_mode('scenario', args.master_mode)['master_profile']
        if args.profile is not None and args.profile != profile:
            parser.error('--profile disagrees with --master-mode')
        args.profile = profile
    else:
        args.profile = args.profile or catalog.default_profile
    if args.profile == 'all':
        args.profile = None


def selection_arguments(args):
    if args.suite is not None:
        return ['--suite', args.suite]
    if args.kind is not None:
        return ['--kind', args.kind]
    return ['--all']
