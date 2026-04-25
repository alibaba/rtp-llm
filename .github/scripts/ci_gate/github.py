"""GitHub REST API helpers."""

from typing import Any, Dict, List

from .common import GITHUB_API, GateError, http_json


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
            "::error::GitHub API rate limited or forbidden (HTTP 403) during %s: %s" % (context, message),
            2,
        )
    if status == 404:
        raise GateError("::error::Not found (HTTP 404) during %s: %s" % (context, message), 2)
    raise GateError("::error::GitHub API returned HTTP %d during %s: %s" % (status, context, message), 2)


def github_get(repo, path, context, github_token):
    # type: (str, str, str, str) -> Any
    status, body, _ = http_json(
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
