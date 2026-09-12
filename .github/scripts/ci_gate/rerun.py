"""Rerun-pr-build subcommand: find and rerun the native PR HEAD build run."""

import argparse
import time

from .common import GateError, log, short_sha
from .github import (
    github_get_pages,
    is_rate_limited,
    list_workflow_runs,
    post_pr_comment,
    rerun_workflow_run,
)

ACTIVE_STATUSES = frozenset([
    "queued", "in_progress", "waiting", "pending", "requested",
])

# Keyed on the HEAD SHA *and* the reason. SHA alone is wrong: one HEAD can
# legitimately produce two different explanations in sequence -- a fork run first
# reports action_required, and after a maintainer approves it and it fails, the
# next LGTM hits the rerun 403 -- and a reason-blind key would suppress the
# second, leaving stale advice on the PR. The run id is deliberately absent: a
# re-push of the same SHA or a close/reopen creates a new run, so including it
# would post one comment per run. The dispatcher also fires on
# `issue_comment: [created, edited]`, so every edit replays this whole path.
DEDUPE_MARKER = "<!-- ci-gate:rerun-blocked sha=%s reason=%s -->"

_NOT_A_REVIEW_PROBLEM = (
    "The review requirement is satisfied — this is a CI plumbing limitation, not "
    "a review problem. **The `build` check still shows its previous result, so "
    "this PR has not been verified by CI.** Do not merge on the strength of a "
    "green dispatcher alone.\n\n"
    "**Do NOT push an empty commit:** the gate matches approvals to the exact "
    "HEAD SHA, so a new commit invalidates the current approval and brings you "
    "right back here.\n\n"
)

NO_RUN_COMMENT = (
    "CI dispatcher could not find a native `build` run for HEAD SHA `%s`.\n\n"
    "This can happen if the PR was opened before the CI architecture change, "
    "or if the original run was deleted.\n\n"
    "**To fix:** push any commit (even empty: "
    "`git commit --allow-empty -m \"trigger CI\" && git push`) "
    "to create a native build run, then re-approve or post `lgtm ready to ci`."
)

EXPIRED_RUN_COMMENT = (
    "CI dispatcher found native `build` run %s for HEAD SHA `%s`, "
    "but it is too old to rerun (>30 days).\n\n"
    "**To fix:** push any commit (even empty: "
    "`git commit --allow-empty -m \"trigger CI\" && git push`) "
    "to create a fresh native build run, then re-approve or post `lgtm ready to ci`."
)

APPROVAL_REQUIRED_FORK_COMMENT = (
    "CI dispatcher cannot start native `build` run %s for HEAD SHA `%s`: the run "
    "is waiting for maintainer approval (`action_required`), which GitHub "
    "requires before it will run a workflow from a fork (`%s`).\n\n"
    + _NOT_A_REVIEW_PROBLEM +
    "**To fix:** a maintainer opens the Actions tab and clicks "
    "**\"Approve and run workflows\"** on run %s. Approval carries their "
    "permissions; the dispatcher's token cannot grant it."
)

APPROVAL_REQUIRED_COMMENT = (
    "CI dispatcher cannot start native `build` run %s for HEAD SHA `%s`: the run "
    "is waiting for approval (`action_required`).\n\n"
    + _NOT_A_REVIEW_PROBLEM +
    "**To fix:** a maintainer opens the Actions tab and approves run %s. "
    "Approval carries their permissions; the dispatcher's token cannot grant it."
)

RERUN_FORBIDDEN_FORK_COMMENT = (
    "CI dispatcher cannot rerun native `build` run %s for HEAD SHA `%s`: the run "
    "belongs to a fork (`%s`), and rerunning a fork's workflow needs the "
    "\"approve and run\" privilege that the default `GITHUB_TOKEN` does not "
    "have.\n\n"
    + _NOT_A_REVIEW_PROBLEM +
    "**To fix:** a maintainer re-runs run %s from the Actions tab, which carries "
    "their permissions.\n\n"
    "_Maintainers: the durable fix is to move the required context off the "
    "App-owned check run — see the FORK LIMITATION note in the dispatcher "
    "workflow header._"
)

RERUN_FORBIDDEN_COMMENT = (
    "CI dispatcher was denied permission to rerun native `build` run %s for HEAD "
    "SHA `%s` (HTTP 403). This is a permission denial, not a rate limit, so "
    "retrying will not help. It is also intermittent: the same branch can rerun "
    "successfully on another attempt.\n\n"
    + _NOT_A_REVIEW_PROBLEM +
    "**To fix:** a maintainer re-runs run %s from the Actions tab, "
    "then re-approve or post `lgtm ready to ci`."
)


def _head_repo(run):
    # type: (dict) -> str
    """Fork detection source. head_repository is null once a fork is deleted."""
    return str((run.get("head_repository") or {}).get("full_name") or "")


def _is_fork(run, repo):
    # type: (dict, str) -> bool
    """Unknown head counts as a fork: null head_repository means it was deleted,
    and the fork wording is the one that stays true in that case."""
    head_repo = _head_repo(run)
    return head_repo.lower() != str(repo).lower()


def _already_commented(repo, pr_number, marker, token):
    # type: (str, str, str, str) -> bool
    """True when this HEAD already carries the marker.

    Any read failure counts as "not found" and we comment anyway. Catching
    everything is deliberate: http_json raises GateError only for URLError, and
    lets socket.timeout, RemoteDisconnected and UnicodeDecodeError through raw.
    Letting any of those escape would re-harden the branch this function exists
    to keep soft, and a duplicate comment is far cheaper than a required check
    stuck red.
    """
    try:
        comments = github_get_pages(
            repo, "/issues/%s/comments" % pr_number,
            "listing comments on PR #%s" % pr_number, token,
        )
    except Exception as exc:
        log("::warning::Could not read existing comments (%s) -- posting anyway" % exc)
        return False
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        if marker in str(comment.get("body") or ""):
            return True
    return False


def _post_once(repo, pr_number, head_sha, reason, body, token):
    # type: (str, str, str, str, str, str) -> None
    """Post a HEAD-and-reason-scoped explanatory comment at most once.

    Never raises, for the same reason _already_commented does not: the caller is
    about to return 0 precisely so a required check does not sit red without an
    explanation, and losing the explanation must not turn that into exit 2.
    """
    marker = DEDUPE_MARKER % (head_sha, reason)
    if _already_commented(repo, pr_number, marker, token):
        log("Explanatory comment already present for HEAD %s (%s)"
            % (short_sha(head_sha), reason))
        return
    try:
        status, response, _ = post_pr_comment(
            repo, pr_number, marker + "\n\n" + body, token)
    except Exception as exc:
        log("::warning::Failed to post PR comment: %s" % exc)
        return
    if status != 201:
        message = response.get("message") if isinstance(response, dict) else response
        # Swallowing this would hide the failure inside a green job.
        log("::warning::Failed to post PR comment (HTTP %d): %s" % (status, message))


def rerun_pr_build(args):
    # type: (argparse.Namespace) -> int
    """Find the native pull_request build run for PR HEAD and rerun it.

    Exit codes:
      0 - rerun triggered, or run already in-progress/succeeded (no action needed)
      0 - rerun impossible for a reason the author cannot fix; PR comment posted
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
        log("No native pull_request run found for HEAD %s -- posting PR comment"
            % short_sha(head_sha))
        _post_once(repo, pr_number, head_sha, "no-run",
                   NO_RUN_COMMENT % short_sha(head_sha), token)
        return 0

    run = runs[0]
    run_id = run.get("id")
    status = (run.get("status") or "").lower()
    conclusion = (run.get("conclusion") or "").lower()
    head_repo = _head_repo(run)
    is_fork = _is_fork(run, repo)
    log("Found run %s (status=%s, conclusion=%s, head_repo=%s, fork=%s)"
        % (run_id, status, conclusion, head_repo or "unknown", is_fork))

    if status in ACTIVE_STATUSES:
        log("Run %s is already active, nothing to do" % run_id)
        return 0

    if conclusion == "success":
        log("Run %s already succeeded, nothing to do" % run_id)
        return 0

    # `action_required` arrives as a *conclusion* alongside status=completed, so
    # it passes both checks above. The run never executed, so rerunning it is a
    # guaranteed 403: what it needs is an approval, not a rerun. A same-repo run
    # can land here too (an environment gate), hence the two wordings.
    if "action_required" in (status, conclusion):
        log("::warning::Run %s for HEAD %s is waiting for approval; "
            "the dispatcher cannot grant it" % (run_id, short_sha(head_sha)))
        if is_fork:
            body_text = APPROVAL_REQUIRED_FORK_COMMENT % (
                run_id, short_sha(head_sha), head_repo or "unknown", run_id)
        else:
            body_text = APPROVAL_REQUIRED_COMMENT % (
                run_id, short_sha(head_sha), run_id)
        _post_once(repo, pr_number, head_sha, "approval", body_text, token)
        return 0

    for attempt in range(1, max_retries + 1):
        http_status, body, response_headers = rerun_workflow_run(repo, run_id, token)

        if http_status in (201, 204):
            log("Rerun triggered for run %s" % run_id)
            return 0

        if http_status == 409:
            log("Run %s transitioned to active (HTTP 409), nothing to do" % run_id)
            return 0

        if http_status == 422:
            log("Run %s too old to rerun (HTTP 422) -- posting PR comment" % run_id)
            _post_once(repo, pr_number, head_sha, "expired",
                       EXPIRED_RUN_COMMENT % (run_id, short_sha(head_sha)), token)
            return 0

        message = body.get("message") if isinstance(body, dict) else str(body)

        if is_rate_limited(http_status, response_headers, body):
            if attempt < max_retries:
                wait = retry_backoff * (2 ** (attempt - 1))
                log("Rate limited (attempt %d/%d), retrying in %.0fs: %s"
                    % (attempt, max_retries, wait, message))
                time.sleep(wait)
                continue
            raise GateError(
                "::error::Rerun failed after %d retries (HTTP %d): %s"
                % (max_retries, http_status, message), 2
            )

        if http_status == 403:
            # A permission denial, not throttling. Retrying cannot help, and
            # failing the job would leave `build` red with no explanation even
            # though the review requirement is satisfied.
            log("::warning::Rerun of run %s denied by permissions: %s"
                % (run_id, message))
            if is_fork:
                body_text = RERUN_FORBIDDEN_FORK_COMMENT % (
                    run_id, short_sha(head_sha), head_repo or "unknown", run_id)
            else:
                body_text = RERUN_FORBIDDEN_COMMENT % (
                    run_id, short_sha(head_sha), run_id)
            _post_once(repo, pr_number, head_sha, "forbidden", body_text, token)
            return 0

        raise GateError(
            "::error::Unexpected HTTP %d rerunning run %s: %s" % (http_status, run_id, message), 2
        )

    raise GateError("::error::Rerun failed after %d retries" % max_retries, 2)
