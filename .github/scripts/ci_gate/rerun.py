"""Rerun-pr-build subcommand: find and rerun the native PR HEAD build run."""

import argparse
import time

from .common import GateError, log, short_sha
from .github import list_workflow_runs, post_pr_comment, rerun_workflow_run

ACTIVE_STATUSES = frozenset([
    "queued", "in_progress", "waiting", "pending", "requested",
])

NO_RUN_COMMENT = (
    "CI dispatcher could not find a native `build` run for HEAD SHA `%s`.\n\n"
    "This can happen if the PR was opened before the CI architecture change, "
    "or if the original run was deleted.\n\n"
    "**To fix:** push any commit (even empty: "
    "`git commit --allow-empty -m \"trigger CI\" && git push`) "
    "to create a native build run, then re-approve or post `lgtm ready to ci`."
)

EXPIRED_RUN_COMMENT = (
    "CI dispatcher found native `build` run %d for HEAD SHA `%s`, "
    "but it is too old to rerun (>30 days).\n\n"
    "**To fix:** push any commit (even empty: "
    "`git commit --allow-empty -m \"trigger CI\" && git push`) "
    "to create a fresh native build run, then re-approve or post `lgtm ready to ci`."
)


def rerun_pr_build(args):
    # type: (argparse.Namespace) -> int
    """Find the native pull_request build run for PR HEAD and rerun it.

    Exit codes:
      0 - rerun triggered, or run already in-progress/succeeded (no action needed)
      0 - no native run found, but actionable PR comment posted (soft failure)
      2 - API error (retries exhausted)
    """
    repo = args.repository
    pr_number = args.pr_number
    head_sha = args.head_sha
    workflow_file = args.workflow_file
    token = args.github_token
    max_retries = args.max_retries
    retry_backoff = args.retry_backoff

    runs = list_workflow_runs(repo, workflow_file, "pull_request", head_sha, token)

    if not runs:
        log("No native pull_request run found for HEAD %s — posting PR comment" % short_sha(head_sha))
        post_pr_comment(repo, pr_number, NO_RUN_COMMENT % short_sha(head_sha), token)
        return 0

    run = runs[0]
    run_id = run.get("id")
    status = (run.get("status") or "").lower()
    conclusion = (run.get("conclusion") or "").lower()
    log("Found run %s (status=%s, conclusion=%s)" % (run_id, status, conclusion))

    if status in ACTIVE_STATUSES:
        log("Run %s is already active, nothing to do" % run_id)
        return 0

    if conclusion == "success":
        log("Run %s already succeeded, nothing to do" % run_id)
        return 0

    for attempt in range(1, max_retries + 1):
        http_status, body = rerun_workflow_run(repo, run_id, token)

        if http_status in (201, 204):
            log("Rerun triggered for run %s" % run_id)
            return 0

        if http_status == 409:
            log("Run %s transitioned to active (HTTP 409), nothing to do" % run_id)
            return 0

        if http_status == 422:
            log("Run %s too old to rerun (HTTP 422) — posting PR comment" % run_id)
            post_pr_comment(
                repo, pr_number,
                EXPIRED_RUN_COMMENT % (run_id, short_sha(head_sha)),
                token,
            )
            return 0

        if http_status == 403:
            message = body.get("message") if isinstance(body, dict) else str(body)
            if attempt < max_retries:
                wait = retry_backoff * (2 ** (attempt - 1))
                log("Rate limited (attempt %d/%d), retrying in %.0fs: %s"
                    % (attempt, max_retries, wait, message))
                time.sleep(wait)
                continue
            raise GateError(
                "::error::Rerun failed after %d retries (HTTP 403): %s" % (max_retries, message), 2
            )

        message = body.get("message") if isinstance(body, dict) else str(body)
        raise GateError(
            "::error::Unexpected HTTP %d rerunning run %s: %s" % (http_status, run_id, message), 2
        )

    raise GateError("::error::Rerun failed after %d retries" % max_retries, 2)
