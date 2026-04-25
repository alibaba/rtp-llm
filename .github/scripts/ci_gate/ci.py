"""CI status checking, waiting, and triggering subcommands."""

import argparse
import time

from .common import BRANCH_REF, CI_TRIGGER_URL, PIPELINE_ID, PROJECT_ID, GateError, log, write_output
from .ci_service import ci_service_request, get_branch_info, parse_ci_status, retrieve_task_status


def _write_pre_check_action(args, action):
    # type: (argparse.Namespace, str) -> None
    write_output("ci_action", action, getattr(args, "output_file", ""))


def pre_check_status(args):
    # type: (argparse.Namespace) -> int
    max_attempts = args.max_attempts
    sleep_interval = args.sleep_interval
    main_status = "UNKNOWN"
    log("Pre-checking CI status for commitId: %s ..." % args.commit_id)
    log("Will check %d times with %ds interval" % (max_attempts, sleep_interval))

    for attempt in range(1, max_attempts + 1):
        log("")
        log("=== Attempt %d/%d ===" % (attempt, max_attempts))
        try:
            response = retrieve_task_status(args.commit_id, args.security, args.repository)
        except GateError as exc:
            log(str(exc))
            if attempt < max_attempts:
                log("Will retry in %d seconds..." % sleep_interval)
                time.sleep(sleep_interval)
                continue
            log("All attempts failed, need to run CI")
            _write_pre_check_action(args, "trigger")
            return 1

        log("Current commitId: %s, taskId: %s" % (response.get("commitId"), response.get("taskId")))
        main_status, status_summary = parse_ci_status(response)
        log("Current status: %s" % status_summary)
        log("Current main status: %s" % main_status)

        if main_status == "DONE":
            log("CI already completed successfully for this commit")
            log("Skipping CI trigger")
            _write_pre_check_action(args, "done")
            return 0
        if main_status == "FAILED":
            log("CI has already failed or been canceled for this commit")
            log("Will re-trigger CI")
            _write_pre_check_action(args, "trigger")
            return 1
        if attempt < max_attempts:
            log("CI status is %s, will check again in %d seconds..." % (main_status, sleep_interval))
            time.sleep(sleep_interval)

    log("")
    log("=== Final Result ===")
    if main_status in {"RUNNING", "PENDING"}:
        log("CI is %s for this commit after %d checks, skipping trigger but waiting for result" % (main_status, max_attempts))
        _write_pre_check_action(args, "wait")
        return 0
    log("CI status is %s after %d checks, allowing CI trigger" % (main_status, max_attempts))
    _write_pre_check_action(args, "trigger")
    return 1


def wait_status(args):
    # type: (argparse.Namespace) -> int
    max_wait_time = args.max_wait_time
    max_wait_pending_time = args.max_wait_pending_time
    max_wait_running_time = args.max_wait_running_time
    overall_start = time.time()
    running_start = None  # type: float

    while True:
        log("Querying CI status for commitId: %s ..." % args.commit_id)
        response = retrieve_task_status(args.commit_id, args.security, args.repository)
        current_time = time.time()
        overall_elapsed = int(current_time - overall_start)
        if overall_elapsed > max_wait_time:
            raise GateError("Error: Overall timeout waiting for CI completion (waited %d seconds)" % overall_elapsed)

        log("Current commitId: %s, taskId: %s" % (response.get("commitId"), response.get("taskId")))
        main_status, status_summary = parse_ci_status(response)
        log("Current status: %s" % status_summary)
        log("Current main status: %s" % main_status)

        if main_status == "PENDING":
            log("PENDING elapsed: %ds / %ds" % (overall_elapsed, max_wait_pending_time))
            if overall_elapsed > max_wait_pending_time:
                raise GateError("Error: Timeout waiting for task to start (PENDING for %d seconds)" % overall_elapsed)
        if main_status == "RUNNING":
            if running_start is None:
                running_start = current_time
                log("Task started running, begin RUNNING timer")
            running_elapsed = int(current_time - running_start)
            log("RUNNING elapsed: %ds / %ds" % (running_elapsed, max_wait_running_time))
            if running_elapsed > max_wait_running_time:
                raise GateError("Error: Timeout waiting for CI to finish after RUNNING (waited %d seconds)" % running_elapsed)

        if main_status == "DONE":
            log("CI completed successfully")
            return 0
        if main_status == "FAILED":
            task_id = response.get("taskId")
            log(
                "CI failed with commitId: %s, status: %s, task link: "
                "https://code.alibaba-inc.com/foundation_models/RTP-LLM/ci/jobs?"
                "pipelineId=%s&pipelineRunId=%s&createType=yaml" % (args.commit_id, status_summary, PIPELINE_ID, task_id)
            )
            return 1
        time.sleep(20)


def trigger_ci(args):
    # type: (argparse.Namespace) -> int
    github_repository = args.repository
    branch_name = "open_merge/%s" % args.github_pr_id
    try:
        branch_info = get_branch_info(branch_name, github_repository, args.commit_id, args.security)
        current_internal_commit_id = str(((branch_info.get("commit") or {}).get("id")) or "UNKNOWN")
    except GateError as exc:
        raise GateError("::error::Failed to retrieve internal commit id for %s: %s" % (branch_name, exc))
    if current_internal_commit_id == "UNKNOWN":
        raise GateError("::error::Internal commit id is UNKNOWN for branch %s — cannot trigger CI safely" % branch_name)

    payload = {
        "type": "CREATE-TASK",
        "commitId": args.commit_id,
        "currentInternalCommitId": current_internal_commit_id,
        "repositoryUrl": "https://github.com/%s.git" % github_repository,
        "prId": args.github_pr_id,
        "aone": {"projectId": PROJECT_ID, "pipelineId": PIPELINE_ID},
        "newBranch": {"name": branch_name, "ref": BRANCH_REF, "head": "UNKNOWN"},
        "params": {
            "cancel-in-progress": "true",
            "github_commit": args.commit_id,
            "github_source_repo": args.github_source_repo,
            "github_run_id": args.github_run_id,
            "aone_branch_name": branch_name,
            "aone_branch_ref": BRANCH_REF,
        },
    }
    body = ci_service_request(payload, args.security, "triggering CI", CI_TRIGGER_URL)
    if not isinstance(body, dict):
        raise GateError("::error::CI trigger returned non-dict response: %s" % body)
    if body.get("success") is False:
        raise GateError("::error::CI trigger rejected: %s" % (body.get("errorMsg") or body.get("error") or body))
    status = str(body.get("status", "")).upper()
    if status in {"FAILED", "ERROR"}:
        raise GateError("::error::CI trigger failed: %s" % body)
    return 0
