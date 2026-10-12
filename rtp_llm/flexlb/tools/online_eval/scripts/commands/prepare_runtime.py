#!/usr/bin/env python3
"""Prepare the optional ZooKeeper runtime before dispatching HA runs."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from runtime.java_runtime import resolve_java21
from runtime.zk_helper import ZkHelperOps

if __name__ == '__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    directory = ZkHelperOps.prepare(resolve_java21())
    print('ZooKeeper runtime prepared: ' + str(directory))
