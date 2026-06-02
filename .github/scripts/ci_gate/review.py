"""Review qualification and resolve-context subcommand."""

import argparse
from typing import Any, Dict, List

from .common import GateError, is_true, log, short_sha, write_output
from .github import github_get, github_get_pages


def latest_fresh_reviews(reviews, head_sha, pr_author):
    # type: (List[Dict[str, Any]], str, str) -> List[Dict[str, Any]]
    latest_by_user = {}  # type: Dict[str, Dict[str, Any]]
    for review in reviews:
        user = review.get("user") or {}
        login = user.get("login")
        if review.get("commit_id") != head_sha:
            continue
        if user.get("type") == "Bot":
            continue
        if review.get("state") == "COMMENTED" and login == pr_author:
            continue
        if not login:
            continue
        previous = latest_by_user.get(login)
        if previous is None or int(review.get("id", 0)) > int(previous.get("id", 0)):
            latest_by_user[login] = review
    return list(latest_by_user.values())


def _fetch_head_commit_date(repo, head_sha, github_token):
    # type: (str, str, str) -> str
    """Return the committer date (ISO 8601) of the given commit."""
    commit = github_get(
        repo, "/commits/%s" % head_sha,
        "fetching commit %s" % short_sha(head_sha), github_token,
    )
    if not isinstance(commit, dict):
        raise GateError("::error::Unexpected commit response for %s" % short_sha(head_sha), 2)
    return (((commit.get("commit") or {}).get("committer") or {}).get("date")) or ""


def _check_issue_comments_qualified(pr_number, repo, head_sha, github_token, lgtm_user):
    # type: (str, str, str, str, str) -> bool
    """Check whether a fresh LGTM issue comment from *lgtm_user* exists.

    Freshness is defined as ``comment.updated_at >= head_commit.committer.date``
    because issue comments have no ``commit_id`` field.
    """
    head_date = _fetch_head_commit_date(repo, head_sha, github_token)
    if not head_date:
        log("Could not determine head commit date, skipping issue comment check")
        return False

    comments = github_get_pages(
        repo, "/issues/%s/comments" % pr_number,
        "fetching issue comments for PR #%s" % pr_number, github_token,
    )

    lgtm_phrase = "lgtm ready to ci"
    latest_match = None  # type: Any
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        user = (comment.get("user") or {}).get("login", "")
        body = (comment.get("body") or "").lower()
        updated_at = comment.get("updated_at", "")
        if user == lgtm_user and lgtm_phrase in body and updated_at >= head_date:
            if latest_match is None or updated_at > (latest_match.get("updated_at") or ""):
                latest_match = comment

    if latest_match:
        log("PR #%s has fresh LGTM issue comment from %s (updated_at: %s >= commit: %s)"
            % (pr_number, lgtm_user, latest_match.get("updated_at", ""), head_date))
        return True

    log("PR #%s has no qualifying fresh issue comment" % pr_number)
    return False


def check_review_qualified(pr_number, repo, head_sha, github_token, lgtm_user):
    # type: (str, str, str, str, str) -> bool
    pr_data = github_get(repo, "/pulls/%s" % pr_number, "fetching PR #%s" % pr_number, github_token)
    if not isinstance(pr_data, dict):
        raise GateError("::error::Unexpected PR response for #%s" % pr_number, 2)

    pr_author = ((pr_data.get("user") or {}).get("login")) or ""
    reviews = github_get_pages(repo, "/pulls/%s/reviews" % pr_number, "fetching reviews for PR #%s" % pr_number, github_token)
    fresh = latest_fresh_reviews([r for r in reviews if isinstance(r, dict)], head_sha, pr_author)
    log("PR #%s: %d fresh review(s) against %s (author: %s)" % (pr_number, len(fresh), short_sha(head_sha), pr_author))

    change_request = next((r for r in fresh if r.get("state") == "CHANGES_REQUESTED"), None)
    if change_request:
        user = (change_request.get("user") or {}).get("login", "unknown")
        log("PR #%s blocked: %s requested changes" % (pr_number, user))
        return False

    if any(r.get("state") == "APPROVED" for r in fresh):
        log("PR #%s has a latest fresh APPROVED review" % pr_number)
        return True

    lgtm_phrase = "lgtm ready to ci"
    for review in fresh:
        user = (review.get("user") or {}).get("login")
        body = (review.get("body") or "").lower()
        if user == lgtm_user and review.get("state") == "COMMENTED" and lgtm_phrase in body:
            log("PR #%s has latest fresh LGTM from %s" % (pr_number, lgtm_user))
            return True

    if _check_issue_comments_qualified(pr_number, repo, head_sha, github_token, lgtm_user):
        return True

    log("PR #%s has no qualifying fresh review or issue comment" % pr_number)
    return False


def resolve_context(args):
    # type: (argparse.Namespace) -> int
    event_name = args.event_name
    repo = args.repository

    if event_name == "workflow_dispatch":
        pr_number = args.input_pr_number
        head_sha = args.input_head_sha
        pr_data = github_get(repo, "/pulls/%s" % pr_number, "fetching PR #%s" % pr_number, args.github_token)
        if not isinstance(pr_data, dict):
            raise GateError("::error::Unexpected PR response for #%s" % pr_number)

        clone_url = (((pr_data.get("head") or {}).get("repo") or {}).get("clone_url")) or ""
        pr_state = pr_data.get("state")
        actual_head = ((pr_data.get("head") or {}).get("sha")) or ""
        if not clone_url:
            raise GateError("::error::Failed to fetch clone_url for PR #%s (state: %s)" % (pr_number, pr_state))
        if pr_state != "open":
            log("::error::PR #%s is %s — CI will not run" % (pr_number, pr_state))
            write_output("head_sha", head_sha, args.output_file)
            write_output("pr_number", pr_number, args.output_file)
            write_output("clone_url", clone_url, args.output_file)
            write_output("qualified", "false", args.output_file)
            return 1
        if actual_head and actual_head != head_sha:
            raise GateError(
                "::error::workflow_dispatch head_sha %s does not match "
                "current PR HEAD %s" % (short_sha(head_sha), short_sha(actual_head))
            )
    else:
        head_sha = args.event_head_sha
        pr_number = args.event_pr_number
        clone_url = args.event_clone_url
        pr_data = github_get(repo, "/pulls/%s" % pr_number, "fetching PR #%s" % pr_number, args.github_token)
        if not isinstance(pr_data, dict):
            raise GateError("::error::Unexpected PR response for #%s" % pr_number)
        actual_head = ((pr_data.get("head") or {}).get("sha")) or ""

        if not head_sha and actual_head:
            head_sha = actual_head
        if not clone_url:
            clone_url = (((pr_data.get("head") or {}).get("repo") or {}).get("clone_url")) or ""

        if actual_head and actual_head != head_sha:
            log(
                "::error::PR HEAD changed since event (%s -> "
                "%s) — CI will not run (new workflow will handle)" % (short_sha(head_sha), short_sha(actual_head))
            )
            write_output("head_sha", head_sha, args.output_file)
            write_output("pr_number", pr_number, args.output_file)
            write_output("clone_url", clone_url, args.output_file)
            write_output("qualified", "false", args.output_file)
            return 1

    write_output("head_sha", head_sha, args.output_file)
    write_output("pr_number", pr_number, args.output_file)
    write_output("clone_url", clone_url, args.output_file)

    if event_name == "workflow_dispatch" and is_true(args.input_skip_review):
        log("::warning::Review check skipped (maintainer override)")
        write_output("qualified", "true", args.output_file)
        return 0

    qualified = check_review_qualified(pr_number, repo, head_sha, args.github_token, args.lgtm_user)
    write_output("qualified", "true" if qualified else "false", args.output_file)
    if not qualified:
        log("::error::No qualifying review — build check will report failure")
        return 1
    return 0
