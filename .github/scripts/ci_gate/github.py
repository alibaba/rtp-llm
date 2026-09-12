"""GitHub REST API helpers."""

from typing import Any, Dict, List, Tuple

from .common import GITHUB_API, GateError, http_json, log


def github_headers(github_token):
    # type: (str) -> Dict[str, str]
    return {
        "Authorization": "token %s" % github_token,
        "Accept": "application/vnd.github+json",
    }


def check_github_status(status, body, context):
    # type: (int, Any, str) -> None
    if status == 200:
        return
    message = body.get("message") if isinstance(body, dict) else str(body)
    if status == 403:
        raise GateError(
            "::error::GitHub API denied the request (HTTP 403) during %s: %s" % (context, message),
            2,
        )
    if status == 404:
        raise GateError("::error::Not found (HTTP 404) during %s: %s" % (context, message), 2)
    raise GateError("::error::GitHub API returned HTTP %d during %s: %s" % (status, context, message), 2)


def is_rate_limited(status, headers, body):
    # type: (int, Dict[str, str], Any) -> bool
    """True only for genuine throttling, where retrying can succeed.

    Permission is tested first and wins. A 403 saying "Resource not accessible
    by integration" is permanent: retrying burns the backoff and then reports
    the wrong cause. Without that short-circuit, a fork rerun 403 that happened
    to arrive alongside x-ratelimit-remaining: 0 would be retried as throttling.
    The subject is left open ("by integration", "by personal access token") so
    the check survives a switch to a fine-grained token.

    Headers come before prose because secondary limits often carry no "rate
    limit" text at all, and abuse detection carries neither that nor a
    Retry-After. x-ratelimit-reset is deliberately not consulted: it rides on
    essentially every authenticated response and is always in the future, so
    keying on it would reclassify every 403 as throttled.
    """
    message = body.get("message") if isinstance(body, dict) else body
    message = str(message or "").lower()
    if "not accessible by" in message:
        return False
    if status == 429:
        return True
    if status != 403:
        return False
    headers = headers or {}
    if str(headers.get("x-ratelimit-remaining", "")).strip() == "0":
        return True
    retry_after = str(headers.get("retry-after", "")).strip()
    if retry_after and retry_after != "0":
        return True
    for phrase in ("rate limit", "submitted too quickly",
                   "abuse detection", "retry your request"):
        if phrase in message:
            return True
    return False


def github_get(repo, path, context, github_token):
    # type: (str, str, str, str) -> Any
    status, body, _, _ = http_json(
        "%s/repos/%s%s" % (GITHUB_API, repo, path),
        headers=github_headers(github_token),
        context=context,
    )
    check_github_status(status, body, context)
    return body


def github_get_pages(repo, path, context, github_token):
    # type: (str, str, str, str) -> List[Any]
    result = []  # type: List[Any]
    page = 1
    while True:
        separator = "&" if "?" in path else "?"
        body = github_get(repo, "%s%sper_page=100&page=%d" % (path, separator, page), "%s page %d" % (context, page), github_token)
        if not isinstance(body, list):
            message = body.get("message", "unknown") if isinstance(body, dict) else str(body)
            raise GateError("::error::Unexpected GitHub response during %s: %s" % (context, message), 2)
        if not body:
            break
        result.extend(body)
        page += 1
    return result


def github_post(repo, path, context, github_token, payload=None):
    # type: (str, str, str, str, Any) -> Tuple[int, Any, Dict[str, str]]
    """POST to GitHub API. Returns (status_code, body, response_headers).

    Headers are part of the result because a 403 here is ambiguous: callers need
    them to tell throttling apart from a permanent permission denial.
    """
    status, body, _, response_headers = http_json(
        "%s/repos/%s%s" % (GITHUB_API, repo, path),
        headers=github_headers(github_token),
        payload=payload,
        context=context,
        method="POST",
    )
    return status, body, response_headers


def list_workflow_runs(repo, workflow_file, event, head_sha, github_token):
    # type: (str, str, str, str, str) -> List[Dict[str, Any]]
    """List workflow runs filtered by event type and head SHA, newest first."""
    path = "/actions/workflows/%s/runs?event=%s&head_sha=%s&per_page=10" % (
        workflow_file, event, head_sha,
    )
    status, body, _, _ = http_json(
        "%s/repos/%s%s" % (GITHUB_API, repo, path),
        headers=github_headers(github_token),
        context="listing workflow runs for %s" % workflow_file,
    )
    if status != 200:
        message = body.get("message") if isinstance(body, dict) else str(body)
        raise GateError(
            "::error::GitHub API returned HTTP %d listing runs: %s" % (status, message), 2
        )
    if not isinstance(body, dict):
        raise GateError("::error::Unexpected response listing workflow runs", 2)
    runs = body.get("workflow_runs") or []
    runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return runs


def rerun_workflow_run(repo, run_id, github_token):
    # type: (str, int, str) -> Tuple[int, Any, Dict[str, str]]
    """Full rerun: POST /repos/{repo}/actions/runs/{run_id}/rerun."""
    return github_post(
        repo, "/actions/runs/%s/rerun" % run_id,
        "rerunning workflow run %s" % run_id, github_token,
    )


def post_pr_comment(repo, pr_number, body_text, github_token):
    # type: (str, str, str, str) -> Tuple[int, Any, Dict[str, str]]
    """POST a comment on a PR (via issues API)."""
    return github_post(
        repo, "/issues/%s/comments" % pr_number,
        "posting comment on PR #%s" % pr_number,
        github_token, payload={"body": body_text},
    )
