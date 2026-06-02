"""CLI entry point for the ci_gate package.

Usage: python3 -m ci_gate <subcommand> [args...]
"""

import argparse
import json
import sys
from typing import List

from .common import GateError, log
from .review import check_review_qualified, resolve_context
from .merge import check_merge_conflicts, trigger_merge, wait_merge
from .ci import pre_check_status, wait_status, trigger_ci
from .comment import sync_comment
from .rerun import rerun_pr_build


def main(argv):
    # type: (List[str]) -> int
    parser = argparse.ArgumentParser(description="GitHub CI gate helpers.")
    subparsers = parser.add_subparsers(dest="command")

    check_review = subparsers.add_parser("check-review")
    check_review.add_argument("pr_number")
    check_review.add_argument("repository")
    check_review.add_argument("head_sha")
    check_review.add_argument("--github-token", required=True)
    check_review.add_argument("--lgtm-user", default="LLLLKKKK")

    resolve = subparsers.add_parser("resolve-context")
    resolve.add_argument("--github-token", required=True)
    resolve.add_argument("--event-name", required=True)
    resolve.add_argument("--repository", required=True)
    resolve.add_argument("--input-pr-number", default="")
    resolve.add_argument("--input-head-sha", default="")
    resolve.add_argument("--input-skip-review", default="false")
    resolve.add_argument("--event-head-sha", default="")
    resolve.add_argument("--event-pr-number", default="")
    resolve.add_argument("--event-clone-url", default="")
    resolve.add_argument("--output-file", default="")
    resolve.add_argument("--lgtm-user", default="LLLLKKKK")

    check_merge_parser = subparsers.add_parser("check-merge")
    check_merge_parser.add_argument("pr_number")
    check_merge_parser.add_argument("commit_id")
    check_merge_parser.add_argument("repository")
    check_merge_parser.add_argument("--github-token", required=True)
    check_merge_parser.add_argument("--max-retries", type=int, default=3)
    check_merge_parser.add_argument("--retry-interval", type=int, default=5)
    check_merge_parser.add_argument("--strict-mergeable", action="store_true")

    pre_check = subparsers.add_parser("pre-check-status")
    pre_check.add_argument("commit_id")
    pre_check.add_argument("security")
    pre_check.add_argument("repository")
    pre_check.add_argument("--max-attempts", type=int, default=6)
    pre_check.add_argument("--sleep-interval", type=int, default=20)
    pre_check.add_argument("--output-file", default="")

    wait = subparsers.add_parser("wait-status")
    wait.add_argument("commit_id")
    wait.add_argument("security")
    wait.add_argument("repository")
    wait.add_argument("--max-wait-time", type=int, default=28800)
    wait.add_argument("--max-wait-pending-time", type=int, default=21600)
    wait.add_argument("--max-wait-running-time", type=int, default=7200)

    trigger = subparsers.add_parser("trigger-ci")
    trigger.add_argument("commit_id")
    trigger.add_argument("security")
    trigger.add_argument("github_source_repo")
    trigger.add_argument("github_pr_id")
    trigger.add_argument("github_run_id")
    trigger.add_argument("--repository", required=True)

    sync_comment_parser = subparsers.add_parser("sync-comment")
    sync_comment_parser.add_argument("pr_id")
    sync_comment_parser.add_argument("repository")
    sync_comment_parser.add_argument("commit_id")
    sync_comment_parser.add_argument("security")
    sync_comment_parser.add_argument("--comment", default="internal source has been updated, please review the changes!")
    sync_comment_parser.add_argument("--source", default="RTP")
    sync_comment_parser.add_argument("--main-branch", default="main-internal")
    sync_comment_parser.add_argument("--max-retries", type=int, default=5)
    sync_comment_parser.add_argument("--retry-interval", type=int, default=5)

    rerun = subparsers.add_parser("rerun-pr-build")
    rerun.add_argument("--repository", required=True)
    rerun.add_argument("--pr-number", required=True)
    rerun.add_argument("--head-sha", required=True)
    rerun.add_argument("--workflow-file", default="CI-request-trigger.yml")
    rerun.add_argument("--github-token", required=True)
    rerun.add_argument("--max-retries", type=int, default=3)
    rerun.add_argument("--retry-backoff", type=float, default=2.0)

    merge_trigger = subparsers.add_parser("trigger-merge")
    merge_trigger.add_argument("commit_id")
    merge_trigger.add_argument("security")
    merge_trigger.add_argument("--repository", required=True)
    merge_trigger.add_argument("--pr-id", required=True)
    merge_trigger.add_argument("--author-email", required=True)
    merge_trigger.add_argument("--author-name", required=True)
    merge_trigger.add_argument("--merge-message", required=True)

    merge_wait = subparsers.add_parser("wait-merge")
    merge_wait.add_argument("commit_id")
    merge_wait.add_argument("security")
    merge_wait.add_argument("repository")
    merge_wait.add_argument("--max-wait-time", type=int, default=7200)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 2
    try:
        if args.command == "check-review":
            return 0 if check_review_qualified(
                args.pr_number, args.repository, args.head_sha, args.github_token, args.lgtm_user
            ) else 1
        if args.command == "resolve-context":
            return resolve_context(args)
        if args.command == "check-merge":
            return check_merge_conflicts(args)
        if args.command == "pre-check-status":
            return pre_check_status(args)
        if args.command == "wait-status":
            return wait_status(args)
        if args.command == "trigger-ci":
            return trigger_ci(args)
        if args.command == "sync-comment":
            return sync_comment(args)
        if args.command == "rerun-pr-build":
            return rerun_pr_build(args)
        if args.command == "trigger-merge":
            return trigger_merge(args)
        if args.command == "wait-merge":
            return wait_merge(args)
    except GateError as exc:
        log(str(exc))
        return exc.exit_code
    except json.JSONDecodeError as exc:
        log("::error::Invalid JSON: %s" % exc)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
