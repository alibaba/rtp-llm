"""Scan-and-trigger orchestrator subcommand."""

import argparse
import time

from .common import GateError, log, short_sha
from .ci import pre_check_status, trigger_ci
from .merge import check_merge_conflicts
from .review import check_review_qualified
from .github import github_get_pages


def scan_and_trigger(args):
    # type: (argparse.Namespace) -> int
    security = args.ci_secret
    max_triggers = args.max_triggers
    trigger_cooldown = args.trigger_cooldown

    prs = github_get_pages(args.repository, "/pulls?state=open&base=main", "fetching open PRs", args.github_token)
    log("Found %d open PRs targeting main" % len(prs))
    if not prs:
        log("No open PRs found, nothing to do")
        return 0

    triggered = 0
    api_errors = 0
    for pr in prs:
        if triggered >= max_triggers:
            log("Reached MAX_TRIGGERS=%d, stopping scan" % max_triggers)
            break
        if not isinstance(pr, dict):
            continue
        pr_number = str(pr.get("number"))
        head_sha = str(((pr.get("head") or {}).get("sha")) or "")
        clone_url = str((((pr.get("head") or {}).get("repo") or {}).get("clone_url")) or "")
        log("")
        log("=== PR #%s (HEAD: %s) ===" % (pr_number, short_sha(head_sha)))

        if pr.get("draft") is True:
            log("  -> Draft PR, skip")
            continue
        if not clone_url or clone_url == "None":
            log("  -> Fork repo deleted (clone_url unavailable), skip")
            continue

        try:
            if not check_review_qualified(pr_number, args.repository, head_sha, args.github_token, args.lgtm_user):
                log("  -> No qualifying fresh review, skip")
                continue
        except GateError as exc:
            log(str(exc))
            log("  -> API error checking review, skip (will retry next scan)")
            api_errors += 1
            continue

        merge_args = argparse.Namespace(
            pr_number=pr_number,
            commit_id=head_sha,
            repository=args.repository,
            github_token=args.github_token,
            max_retries=args.merge_max_retries,
            retry_interval=args.merge_retry_interval,
            strict_mergeable=True,
        )
        try:
            if check_merge_conflicts(merge_args) != 0:
                log("  -> Has merge conflicts or unknown mergeable state, skip")
                continue
        except GateError as exc:
            log(str(exc))
            log("  -> API error checking mergeability, skip (will retry next scan)")
            api_errors += 1
            continue

        pre_args = argparse.Namespace(
            commit_id=head_sha,
            security=security,
            repository=args.repository,
            max_attempts=args.precheck_max_attempts,
            sleep_interval=args.precheck_sleep_interval,
        )
        if pre_check_status(pre_args) == 0:
            log("  -> CI already completed or running, skip")
            continue

        log("  -> Triggering CI for PR #%s" % pr_number)
        trigger_args = argparse.Namespace(
            commit_id=head_sha,
            security=security,
            github_source_repo=clone_url,
            github_pr_id=pr_number,
            github_run_id=args.github_run_id,
            repository=args.repository,
        )
        try:
            trigger_ci(trigger_args)
        except Exception as exc:
            log("  -> Failed to trigger CI for PR #%s: %s" % (pr_number, exc))
            api_errors += 1
            continue
        triggered += 1
        time.sleep(trigger_cooldown)

    log("")
    log("=== Scan complete: triggered CI for %d/%d PRs (API errors: %d) ===" % (triggered, len(prs), api_errors))
    if api_errors:
        log("::error::%d PR(s) skipped due to API errors" % api_errors)
        return 1
    return 0
