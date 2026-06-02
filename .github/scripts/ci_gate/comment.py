"""Comment syncing and adding subcommands."""

import argparse
import time

from .common import PROJECT_ID, GateError, log
from .ci_service import branch_commit_id, ci_service_request, get_branch_info


def add_comment(
    current_branch,
    pr_id,
    repository,
    comment,
    source,
    main_branch,
    commit_id,
    security,
    max_retries,
    retry_interval,
):
    # type: (str, str, str, str, str, str, str, str, int, int) -> int
    payload = {
        "type": "ADD-COMMENT",
        "aone": {"projectId": PROJECT_ID},
        "currentBranch": current_branch,
        "prId": pr_id,
        "repositoryUrl": repository,
        "commitId": commit_id,
        "comment": comment,
        "source": source,
        "mainBranch": main_branch,
    }

    for attempt in range(1, max_retries + 1):
        log("Adding comment to PR #%s (attempt %d/%d)" % (pr_id, attempt, max_retries))
        body = ci_service_request(payload, security, "adding GitHub comment")
        if not isinstance(body, dict):
            log("Error: Comment response is not a JSON object")
        elif body.get("success") is not True:
            error = body.get("error") or body.get("errorMsg") or "Unknown error"
            if "Branch info not found" in str(error):
                log("Error: Branch info not cached in Redis")
                log("Please run RETRIEVE-BRANCH-INFO for branches: %s and %s first" % (current_branch, main_branch))
                return 1
            log("Error: Failed to add comment - %s (attempt %d)" % (error, attempt))
        else:
            status = body.get("status")
            message = body.get("message") or ""
            if status == "SUCCESS":
                log("Comment added successfully")
                log("  Current branch (%s) commit: %s" % (current_branch, body.get("currentCommitId")))
                log("  Main branch (%s) commit: %s" % (main_branch, body.get("mainCommitId")))
                return 0
            if status == "SKIPPED":
                log("Comment skipped: %s" % message)
                log("  Both branches have the same commit ID: %s" % body.get("commitId"))
                return 0
            if status == "FAILED":
                log("Error: %s (attempt %d)" % (body.get("error"), attempt))
            else:
                log("Unknown comment status: %s (attempt %d)" % (status, attempt))

        if attempt < max_retries:
            log("Retrying in %ds..." % retry_interval)
            time.sleep(retry_interval)

    log("Error: Failed to add comment after %d attempts" % max_retries)
    return 1


def sync_comment(args):
    # type: (argparse.Namespace) -> int
    current_branch = "open_merge/%s" % args.pr_id
    log("================================================")
    log("Adding GitHub Comment")
    log("================================================")
    log("Repository: %s" % args.repository)
    log("PR ID: %s" % args.pr_id)
    log("Current Branch: %s" % current_branch)
    log("Main Branch: %s" % args.main_branch)
    log("Source: %s" % args.source)
    log("================================================")

    try:
        current_info = get_branch_info(current_branch, args.repository, args.commit_id, args.security)
    except GateError as exc:
        log("::error::Failed to query branch info for %s: %s" % (current_branch, exc))
        return 1

    current_commit_id = branch_commit_id(current_info)
    if not current_commit_id:
        log("::error::Failed to extract commit ID from branch info for %s" % current_branch)
        return 1
    log("Current internal commit ID: %s" % current_commit_id)

    try:
        main_info = get_branch_info(args.main_branch, args.repository, args.commit_id, args.security)
    except GateError as exc:
        log("::error::Failed to query branch info for %s: %s" % (args.main_branch, exc))
        return 1

    main_commit_id = branch_commit_id(main_info)
    if not main_commit_id:
        log("::error::Failed to extract commit ID from main branch info for %s" % args.main_branch)
        return 1
    log("Main internal commit ID: %s" % main_commit_id)

    if current_commit_id == main_commit_id:
        log("Commit IDs are the same, no comment needed")
        return 0

    log("Commit IDs differ, adding comment...")
    return add_comment(
        current_branch=current_branch,
        pr_id=args.pr_id,
        repository=args.repository,
        comment=args.comment,
        source=args.source,
        main_branch=args.main_branch,
        commit_id=args.commit_id,
        security=args.security,
        max_retries=args.max_retries,
        retry_interval=args.retry_interval,
    )
