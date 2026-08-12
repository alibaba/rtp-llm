"""Review qualification and resolve-context subcommand."""

import argparse
from typing import Any, Dict, List

from .common import GateError, is_true, log, short_sha, write_output
from .github import github_get, github_get_pages


TRUSTED_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})


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
        # Anyone can review a public repo; only repo/org insiders may gate CI.
        association = (review.get("author_association") or "").upper()
        if association not in TRUSTED_ASSOCIATIONS:
            log("Ignoring %s review from %s (association: %s)"
                % (review.get("state", "?"), login, association or "NONE"))
            continue
        if review.get("state") == "COMMENTED" and login == pr_author:
            continue
        if not login:
            continue
        previous = latest_by_user.get(login)
        if previous is None or int(review.get("id", 0)) > int(previous.get("id", 0)):
            latest_by_user[login] = review
    return list(latest_by_user.values())


def parse_lgtm_users(lgtm_user):
    # type: (str) -> set
    """Parse a comma-separated ``--lgtm-user`` value into a set of logins."""
    return {u.strip() for u in (lgtm_user or "").split(",") if u.strip()}


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

    lgtm_users = parse_lgtm_users(lgtm_user)
    lgtm_phrase = "lgtm ready to ci"
    for review in fresh:
        user = (review.get("user") or {}).get("login")
        body = (review.get("body") or "").lower()
        if user in lgtm_users and review.get("state") == "COMMENTED" and lgtm_phrase in body:
            log("PR #%s has latest fresh LGTM from %s" % (pr_number, user))
            return True

    log("PR #%s has no qualifying fresh review" % pr_number)
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
