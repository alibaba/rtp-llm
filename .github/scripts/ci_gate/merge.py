"""Merge conflict checking, trigger-merge, and wait-merge subcommands."""

import argparse
import json
import time

from .common import BRANCH_REF, CI_TRIGGER_URL, PROJECT_ID, GateError, log
from .ci_service import ci_service_request
from .github import github_get


def check_merge_conflicts(args):
    # type: (argparse.Namespace) -> int
    max_retries = args.max_retries
    retry_interval = args.retry_interval
    strict = args.strict_mergeable
    mergeable = None

    for attempt in range(1, max_retries + 1):
        pr_data = github_get(args.repository, "/pulls/%s" % args.pr_number, "fetching PR #%s" % args.pr_number, args.github_token)
        if not isinstance(pr_data, dict):
            raise GateError("::error::Unexpected PR response for #%s" % args.pr_number, 2)
        mergeable = pr_data.get("mergeable")
        if mergeable is None:
            log("Mergeable status not yet computed (attempt %d/%d), retrying..." % (attempt, max_retries))
            time.sleep(retry_interval)
            continue
        break

    if mergeable is False:
        log("::error::PR #%s has merge conflicts with main. Please rebase or resolve conflicts." % args.pr_number)
        log("")
        log("The PR branch has textual merge conflicts with the main branch.")
        log("Run: git fetch origin main && git rebase origin/main")
        return 1
    if mergeable is None:
        if strict:
            log("::warning::Mergeable status unknown after %d retries, strict mode: blocking" % max_retries)
            return 1
        log("::warning::GitHub could not determine mergeable status for PR #%s after %d attempts, proceeding anyway" % (args.pr_number, max_retries))

    compare = github_get(args.repository, "/compare/%s...main" % args.commit_id, "checking PR staleness", args.github_token)
    if isinstance(compare, dict):
        main_ahead_by = int(compare.get("ahead_by") or 0)
        if main_ahead_by > 0:
            log("::warning::PR #%s is %d commit(s) behind main (no conflicts, CI will proceed)" % (args.pr_number, main_ahead_by))
        else:
            log("PR #%s is up to date with main" % args.pr_number)
    return 0


def trigger_merge(args):
    # type: (argparse.Namespace) -> int
    repo_url = "https://github.com/%s.git" % args.repository
    branch_name = "open_merge/%s" % args.pr_id
    payload = {
        "type": "MERGE-TASK",
        "repositoryUrl": repo_url,
        "commitId": args.commit_id,
        "prId": args.pr_id,
        "aone": {"projectId": PROJECT_ID},
        "authorEmail": args.author_email,
        "authorName": args.author_name,
        "mergeMessage": args.merge_message,
        "mergeType": "REBASE",
        "sourceBranch": branch_name,
        "targetBranch": BRANCH_REF,
        "actions": {"deleteSourceBranch": False},
    }
    log("Sending MERGE-TASK for commitId: %s" % args.commit_id)
    body = ci_service_request(payload, args.security, "triggering merge", CI_TRIGGER_URL)
    if not isinstance(body, dict):
        raise GateError("::error::Merge trigger returned non-dict response: %s" % body)
    if body.get("success") is False:
        raise GateError("::error::Merge trigger rejected: %s" % (body.get("errorMsg") or body.get("error") or body))
    status = str(body.get("status", "")).upper()
    if status in {"FAILED", "ERROR"}:
        raise GateError("::error::Merge trigger failed: %s" % body)
    return 0


def wait_merge(args):
    # type: (argparse.Namespace) -> int
    max_wait_time = args.max_wait_time
    overall_start = time.time()

    while True:
        log("Querying merge status for commitId: %s ..." % args.commit_id)
        body = ci_service_request(
            {
                "type": "RETRIEVE-MERGE-STATUS",
                "repositoryUrl": args.repository,
                "commitId": args.commit_id,
            },
            args.security,
            "querying merge status",
        )

        overall_elapsed = int(time.time() - overall_start)
        if overall_elapsed > max_wait_time:
            raise GateError("Error: Timeout waiting for merge completion (waited %d seconds)" % overall_elapsed)

        if not isinstance(body, dict):
            raise GateError("::error::Merge status response is not a JSON object")

        status = body.get("status")
        if status == "PENDING":
            log("Merge is still pending...")
            time.sleep(5)
            continue

        if isinstance(status, dict):
            success = status.get("success")
        elif isinstance(status, str):
            try:
                parsed = json.loads(status)
                success = parsed.get("success") if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                success = None
        else:
            success = None

        log("Merge completed with status: %s" % success)
        log("Response: %s" % json.dumps(body, separators=(",", ":")))

        if success is True or str(success).lower() == "true":
            return 0
        raise GateError("Merge failed: %s" % body)
