#!/opt/conda310/bin/python

import argparse
import datetime
import os
import sys
import time
from pathlib import Path


class SuezOpsPackage:
    def __init__(self, package_url, version, emp_id, environment):
        self._package_url = package_url
        self._version = version
        self._emp_id = emp_id
        self._environment = environment

    def create(self):
        sys.path.insert(
            0,
            str(Path(__file__).resolve().parents[3] /
                "package/script/suezops_package"))
        from suez_sdk.suezops_sdk import SuezOpsSDK

        sdk = SuezOpsSDK(self._emp_id, False, self._environment)
        package_info = self._package_info()
        for attempt in range(4):
            try:
                result = sdk.create_online_package(package_info)
                if "id" in result:
                    package_name = result.get(
                        "package_name", package_info["package_name"])
                    print("SUEZOPS_PACKAGE_NAME={}".format(package_name))
                    return result
                print("SuezOps package response:", result)
            except Exception as error:
                print("create SuezOps package failed:", error)
            if attempt < 3:
                time.sleep(1)
        raise RuntimeError("failed to create SuezOps package")

    def package_info(self):
        return self._package_info()

    def _package_info(self):
        version_parts = self._version.split(".")
        if len(version_parts) != 3 or not all(version_parts):
            raise ValueError("version must use major.minor.patch")
        release_tag = "test" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        return {
            "package_name": "navi_sched_simulate_test_{}-{}".format(
                self._version, release_tag),
            "scope": "test",
            "platform": "RTP",
            "release_tag": release_tag,
            "major_version": ".".join(version_parts[:2]),
            "minor_version": version_parts[2],
            "package_json": [{"packageURI": self._package_url, "type": "ARCHIVE"}],
            "release_stage": "BETA",
            "team_name": "suezops",
            "allow_admin_use": True,
            "description": "NaviSched simulator package",
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-url", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--emp-id", required=True)
    parser.add_argument("--environment", default=os.environ.get("SUEZOPS_ENV", "online"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    package = SuezOpsPackage(
        args.package_url, args.version, args.emp_id, args.environment)
    if args.dry_run:
        print(package.package_info())
        return
    package.create()


if __name__ == "__main__":
    main()
