#!/usr/bin/env python3
"""Compare real captures with production synthetic traffic; write standalone HTML/MD/JSON."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from traffic.datasets import profile_path
from analysis.traffic_fidelity import run
from analysis.traffic_fidelity_report import write, markdown


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,default=profile_path())
    parser.add_argument('--captures',type=Path,nargs='*',help='default: all bundled captures; empty: self-consistency only')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--count',type=int,help='generated requests per capture; default: capture event count')
    parser.add_argument('--refit',action='store_true',help='fit each capture separately for in-sample method comparison')
    parser.add_argument('--thresholds',type=Path,help='JSON metric -> [pass_max,warn_max]')
    parser.add_argument('--check',action='store_true',help='exit 2 if source identity/parameter audit is unavailable or mismatched')
    args=parser.parse_args(argv)
    if args.count is not None and not 1<=args.count<=1000000:
        parser.error('--count must be between 1 and 1000000')
    result=run(args.profile,args.captures,seed=args.seed,count=args.count,refit=args.refit,
               limits=json.loads(args.thresholds.read_text()) if args.thresholds else None,
               progress=lambda message: print(message,file=sys.stderr,flush=True))
    write(result,args.out)
    print(markdown(result))
    print(args.out/'fidelity.html')
    return 2 if args.check and (result['identity']['status']!='OK' or any(r['audit']['status']!='OK' for r in result['rows'])) else 0


if __name__=='__main__':
    raise SystemExit(main())
