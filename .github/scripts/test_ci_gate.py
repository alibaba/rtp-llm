"""Unit tests for ci_gate package (mock HTTP, no network calls)."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from ci_gate.common import GateError, is_true
from ci_gate.ci_service import collect_status_tokens, parse_ci_status
from ci_gate.review import (
    _check_issue_comments_qualified,
    _fetch_head_commit_date,
    check_review_qualified,
    latest_fresh_reviews,
    resolve_context,
)
from ci_gate.ci import pre_check_status, trigger_ci, wait_status
from ci_gate.merge import check_merge_conflicts, trigger_merge, wait_merge


# ---------------------------------------------------------------------------
# common.is_true
# ---------------------------------------------------------------------------
class TestIsTrue(unittest.TestCase):
    def test_true_string(self):
        self.assertTrue(is_true("true"))
        self.assertTrue(is_true("True"))
        self.assertTrue(is_true("TRUE"))

    def test_false_string(self):
        self.assertFalse(is_true("false"))
        self.assertFalse(is_true("False"))
        self.assertFalse(is_true(""))

    def test_bool(self):
        self.assertTrue(is_true(True))
        self.assertFalse(is_true(False))

    def test_none(self):
        self.assertFalse(is_true(None))


# ---------------------------------------------------------------------------
# ci_service.collect_status_tokens
# ---------------------------------------------------------------------------
class TestCollectStatusTokens(unittest.TestCase):
    def test_plain_string(self):
        self.assertEqual(collect_status_tokens("SUCCESS"), ["SUCCESS"])

    def test_none(self):
        self.assertEqual(collect_status_tokens(None), [])

    def test_dict_with_status_key(self):
        self.assertEqual(collect_status_tokens({"status": "DONE"}), ["DONE"])

    def test_nested_dict(self):
        tokens = collect_status_tokens({"outer": {"status": "RUNNING"}})
        self.assertIn("RUNNING", tokens)

    def test_list(self):
        tokens = collect_status_tokens(["SUCCESS", "DONE"])
        self.assertEqual(tokens, ["SUCCESS", "DONE"])

    def test_status_map(self):
        tokens = collect_status_tokens(
            {"job1": "SUCCESS", "job2": "FAILED"}, status_map=True
        )
        self.assertIn("SUCCESS", tokens)
        self.assertIn("FAILED", tokens)


# ---------------------------------------------------------------------------
# ci_service.parse_ci_status
# ---------------------------------------------------------------------------
class TestParseCiStatus(unittest.TestCase):
    def test_done(self):
        status, _ = parse_ci_status({"status": "SUCCESS"})
        self.assertEqual(status, "DONE")

    def test_failed(self):
        status, _ = parse_ci_status({"status": "FAILED"})
        self.assertEqual(status, "FAILED")

    def test_running(self):
        status, _ = parse_ci_status({"status": "RUNNING"})
        self.assertEqual(status, "RUNNING")

    def test_pending(self):
        status, _ = parse_ci_status({"status": "PENDING"})
        self.assertEqual(status, "PENDING")

    def test_unknown_null(self):
        status, _ = parse_ci_status({"status": None})
        self.assertEqual(status, "UNKNOWN")

    def test_nested_json_string(self):
        inner = json.dumps({"status": "DONE"})
        status, _ = parse_ci_status({"status": inner})
        self.assertEqual(status, "DONE")

    def test_complex_status_map(self):
        status, _ = parse_ci_status(
            {"status": {"job1": {"status": "SUCCESS"}, "job2": {"status": "SUCCESS"}}}
        )
        self.assertEqual(status, "DONE")

    def test_mixed_status_map_with_failure(self):
        status, _ = parse_ci_status(
            {"status": {"job1": {"status": "SUCCESS"}, "job2": {"status": "FAILED"}}}
        )
        self.assertEqual(status, "FAILED")

    def test_timeout_status(self):
        status, _ = parse_ci_status({"status": "TIMEOUT"})
        self.assertEqual(status, "FAILED")

    def test_canceled_status(self):
        status, _ = parse_ci_status({"status": "CANCELED"})
        self.assertEqual(status, "FAILED")

    def test_all_not_run_is_failed(self):
        status, _ = parse_ci_status({"status": "NOT_RUN"})
        self.assertEqual(status, "FAILED")

    def test_all_skipped_is_failed(self):
        status, _ = parse_ci_status({"status": "SKIPPED"})
        self.assertEqual(status, "FAILED")

    def test_all_jobs_not_run_map_is_failed(self):
        status, _ = parse_ci_status(
            {"status": {"job1": {"status": "NOT_RUN"}, "job2": {"status": "SKIPPED"}}}
        )
        self.assertEqual(status, "FAILED")

    def test_success_with_skipped_is_done(self):
        status, _ = parse_ci_status(
            {"status": {"job1": {"status": "SUCCESS"}, "job2": {"status": "SKIPPED"}}}
        )
        self.assertEqual(status, "DONE")

    def test_success_with_not_run_is_done(self):
        status, _ = parse_ci_status(
            {"status": {"job1": {"status": "SUCCESS"}, "job2": {"status": "NOT_RUN"}}}
        )
        self.assertEqual(status, "DONE")


# ---------------------------------------------------------------------------
# review.latest_fresh_reviews
# ---------------------------------------------------------------------------
class TestLatestFreshReviews(unittest.TestCase):
    def _review(self, login, state, commit_id="abc123", user_type="User", review_id=1, body=""):
        return {
            "id": review_id,
            "user": {"login": login, "type": user_type},
            "state": state,
            "commit_id": commit_id,
            "body": body,
        }

    def test_filters_wrong_commit(self):
        reviews = [self._review("alice", "APPROVED", commit_id="other")]
        result = latest_fresh_reviews(reviews, "abc123", "author")
        self.assertEqual(result, [])

    def test_filters_bot(self):
        reviews = [self._review("bot", "APPROVED", user_type="Bot")]
        result = latest_fresh_reviews(reviews, "abc123", "author")
        self.assertEqual(result, [])

    def test_filters_author_comment(self):
        reviews = [self._review("author", "COMMENTED")]
        result = latest_fresh_reviews(reviews, "abc123", "author")
        self.assertEqual(result, [])

    def test_keeps_latest_per_user(self):
        reviews = [
            self._review("alice", "COMMENTED", review_id=1),
            self._review("alice", "APPROVED", review_id=2),
        ]
        result = latest_fresh_reviews(reviews, "abc123", "author")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["state"], "APPROVED")

    def test_multiple_users(self):
        reviews = [
            self._review("alice", "APPROVED", review_id=1),
            self._review("bob", "CHANGES_REQUESTED", review_id=2),
        ]
        result = latest_fresh_reviews(reviews, "abc123", "author")
        self.assertEqual(len(result), 2)


# ---------------------------------------------------------------------------
# review.check_review_qualified (mocked)
# ---------------------------------------------------------------------------
class TestCheckReviewQualified(unittest.TestCase):
    def _mock_pr(self, author="author"):
        return {"user": {"login": author}}

    def _mock_reviews(self, reviews):
        return reviews

    @patch("ci_gate.review.github_get_pages")
    @patch("ci_gate.review.github_get")
    def test_approved(self, mock_get, mock_pages):
        mock_get.return_value = self._mock_pr()
        mock_pages.return_value = [
            {"id": 1, "user": {"login": "reviewer", "type": "User"},
             "state": "APPROVED", "commit_id": "sha1", "body": ""}
        ]
        result = check_review_qualified("1", "repo", "sha1", "token", "LLLLKKKK")
        self.assertTrue(result)

    @patch("ci_gate.review.github_get_pages")
    @patch("ci_gate.review.github_get")
    def test_changes_requested(self, mock_get, mock_pages):
        mock_get.return_value = self._mock_pr()
        mock_pages.return_value = [
            {"id": 1, "user": {"login": "reviewer", "type": "User"},
             "state": "CHANGES_REQUESTED", "commit_id": "sha1", "body": ""}
        ]
        result = check_review_qualified("1", "repo", "sha1", "token", "LLLLKKKK")
        self.assertFalse(result)

    @patch("ci_gate.review.github_get_pages")
    @patch("ci_gate.review.github_get")
    def test_lgtm_comment(self, mock_get, mock_pages):
        mock_get.return_value = self._mock_pr()
        mock_pages.return_value = [
            {"id": 1, "user": {"login": "LLLLKKKK", "type": "User"},
             "state": "COMMENTED", "commit_id": "sha1",
             "body": "lgtm ready to ci please"}
        ]
        result = check_review_qualified("1", "repo", "sha1", "token", "LLLLKKKK")
        self.assertTrue(result)

    @patch("ci_gate.review.github_get_pages")
    @patch("ci_gate.review.github_get")
    def test_no_reviews(self, mock_get, mock_pages):
        mock_get.return_value = self._mock_pr()
        mock_pages.return_value = []
        result = check_review_qualified("1", "repo", "sha1", "token", "LLLLKKKK")
        self.assertFalse(result)


# ---------------------------------------------------------------------------
# review._check_issue_comments_qualified (mocked)
# ---------------------------------------------------------------------------
class TestCheckIssueCommentsQualified(unittest.TestCase):
    COMMIT_DATE = "2025-04-20T10:00:00Z"
    COMMIT_RESPONSE = {
        "commit": {"committer": {"date": COMMIT_DATE}},
    }

    def _comment(self, login="LLLLKKKK", body="lgtm ready to ci", updated_at="2025-04-20T12:00:00Z"):
        return {"user": {"login": login}, "body": body, "updated_at": updated_at}

    @patch("ci_gate.review.github_get_pages")
    @patch("ci_gate.review.github_get")
    def test_fresh_lgtm_qualifies(self, mock_get, mock_pages):
        mock_get.return_value = self.COMMIT_RESPONSE
        mock_pages.return_value = [self._comment()]
        result = _check_issue_comments_qualified("1", "repo", "sha1", "token", "LLLLKKKK")
        self.assertTrue(result)

    @patch("ci_gate.review.github_get_pages")
    @patch("ci_gate.review.github_get")
    def test_stale_comment_rejected(self, mock_get, mock_pages):
        mock_get.return_value = self.COMMIT_RESPONSE
        mock_pages.return_value = [self._comment(updated_at="2025-04-19T09:00:00Z")]
        result = _check_issue_comments_qualified("1", "repo", "sha1", "token", "LLLLKKKK")
        self.assertFalse(result)

    @patch("ci_gate.review.github_get_pages")
    @patch("ci_gate.review.github_get")
    def test_wrong_author_rejected(self, mock_get, mock_pages):
        mock_get.return_value = self.COMMIT_RESPONSE
        mock_pages.return_value = [self._comment(login="other-user")]
        result = _check_issue_comments_qualified("1", "repo", "sha1", "token", "LLLLKKKK")
        self.assertFalse(result)

    @patch("ci_gate.review.github_get_pages")
    @patch("ci_gate.review.github_get")
    def test_missing_phrase_rejected(self, mock_get, mock_pages):
        mock_get.return_value = self.COMMIT_RESPONSE
        mock_pages.return_value = [self._comment(body="looks good")]
        result = _check_issue_comments_qualified("1", "repo", "sha1", "token", "LLLLKKKK")
        self.assertFalse(result)

    @patch("ci_gate.review.github_get_pages")
    @patch("ci_gate.review.github_get")
    def test_no_comments(self, mock_get, mock_pages):
        mock_get.return_value = self.COMMIT_RESPONSE
        mock_pages.return_value = []
        result = _check_issue_comments_qualified("1", "repo", "sha1", "token", "LLLLKKKK")
        self.assertFalse(result)

    @patch("ci_gate.review.github_get_pages")
    @patch("ci_gate.review.github_get")
    def test_case_insensitive_phrase(self, mock_get, mock_pages):
        mock_get.return_value = self.COMMIT_RESPONSE
        mock_pages.return_value = [self._comment(body="LGTM Ready To CI")]
        result = _check_issue_comments_qualified("1", "repo", "sha1", "token", "LLLLKKKK")
        self.assertTrue(result)

    @patch("ci_gate.review.github_get_pages")
    @patch("ci_gate.review.github_get")
    def test_picks_latest_matching_comment(self, mock_get, mock_pages):
        mock_get.return_value = self.COMMIT_RESPONSE
        mock_pages.return_value = [
            self._comment(updated_at="2025-04-20T11:00:00Z"),
            self._comment(updated_at="2025-04-20T14:00:00Z"),
        ]
        result = _check_issue_comments_qualified("1", "repo", "sha1", "token", "LLLLKKKK")
        self.assertTrue(result)


# ---------------------------------------------------------------------------
# review.check_review_qualified — issue comment fallback
# ---------------------------------------------------------------------------
class TestCheckReviewQualifiedWithIssueComments(unittest.TestCase):
    """Verify that check_review_qualified falls back to issue comments."""

    @patch("ci_gate.review._check_issue_comments_qualified")
    @patch("ci_gate.review.github_get_pages")
    @patch("ci_gate.review.github_get")
    def test_no_reviews_falls_back_to_issue_comments(self, mock_get, mock_pages, mock_issue):
        mock_get.return_value = {"user": {"login": "author"}}
        mock_pages.return_value = []
        mock_issue.return_value = True
        result = check_review_qualified("1", "repo", "sha1", "token", "LLLLKKKK")
        self.assertTrue(result)
        mock_issue.assert_called_once_with("1", "repo", "sha1", "token", "LLLLKKKK")

    @patch("ci_gate.review._check_issue_comments_qualified")
    @patch("ci_gate.review.github_get_pages")
    @patch("ci_gate.review.github_get")
    def test_approved_review_skips_issue_comments(self, mock_get, mock_pages, mock_issue):
        mock_get.return_value = {"user": {"login": "author"}}
        mock_pages.return_value = [
            {"id": 1, "user": {"login": "rev", "type": "User"},
             "state": "APPROVED", "commit_id": "sha1", "body": ""}
        ]
        result = check_review_qualified("1", "repo", "sha1", "token", "LLLLKKKK")
        self.assertTrue(result)
        mock_issue.assert_not_called()

    @patch("ci_gate.review._check_issue_comments_qualified")
    @patch("ci_gate.review.github_get_pages")
    @patch("ci_gate.review.github_get")
    def test_changes_requested_blocks_even_with_issue_comment(self, mock_get, mock_pages, mock_issue):
        mock_get.return_value = {"user": {"login": "author"}}
        mock_pages.return_value = [
            {"id": 1, "user": {"login": "rev", "type": "User"},
             "state": "CHANGES_REQUESTED", "commit_id": "sha1", "body": ""}
        ]
        result = check_review_qualified("1", "repo", "sha1", "token", "LLLLKKKK")
        self.assertFalse(result)
        mock_issue.assert_not_called()


# ---------------------------------------------------------------------------
# review.resolve_context — issue_comment event
# ---------------------------------------------------------------------------
class TestResolveContextIssueComment(unittest.TestCase):
    """Verify resolve_context works for issue_comment events where head_sha
    and clone_url are absent from the event payload."""

    def _base_args(self, **overrides):
        defaults = {
            "event_name": "issue_comment",
            "repository": "org/repo",
            "github_token": "tok",
            "input_pr_number": "",
            "input_head_sha": "",
            "input_skip_review": "false",
            "event_head_sha": "",
            "event_pr_number": "42",
            "event_clone_url": "",
            "lgtm_user": "LLLLKKKK",
            "output_file": "",
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    @patch("ci_gate.review.check_review_qualified")
    @patch("ci_gate.review.github_get")
    def test_issue_comment_resolves_head_from_pr(self, mock_get, mock_qualified):
        mock_get.return_value = {
            "user": {"login": "author"},
            "head": {"sha": "abc111", "repo": {"clone_url": "https://github.com/org/repo.git"}},
            "state": "open",
        }
        mock_qualified.return_value = True
        with tempfile.NamedTemporaryFile(mode="r+") as output:
            result = resolve_context(self._base_args(output_file=output.name))
            output.seek(0)
            contents = output.read()
        self.assertEqual(result, 0)
        self.assertIn("head_sha=abc111", contents)
        self.assertIn("clone_url=https://github.com/org/repo.git", contents)
        self.assertIn("qualified=true", contents)
        mock_qualified.assert_called_once_with("42", "org/repo", "abc111", "tok", "LLLLKKKK")

    @patch("ci_gate.review.check_review_qualified")
    @patch("ci_gate.review.github_get")
    def test_issue_comment_unqualified_returns_1(self, mock_get, mock_qualified):
        mock_get.return_value = {
            "user": {"login": "author"},
            "head": {"sha": "abc111", "repo": {"clone_url": "url"}},
            "state": "open",
        }
        mock_qualified.return_value = False
        with tempfile.NamedTemporaryFile(mode="r+") as output:
            result = resolve_context(self._base_args(output_file=output.name))
            output.seek(0)
            contents = output.read()
        self.assertEqual(result, 1)
        self.assertIn("qualified=false", contents)


# ---------------------------------------------------------------------------
# review.resolve_context — qualified=false must return 1 (not 0!)
# ---------------------------------------------------------------------------
class TestResolveContext(unittest.TestCase):
    """Verify that resolve_context returns 1 (failure) whenever qualified=false,
    so the GitHub Actions job does NOT report SUCCESS for unapproved code."""

    def _base_args(self, **overrides):
        defaults = {
            "event_name": "pull_request",
            "repository": "org/repo",
            "github_token": "tok",
            "input_pr_number": "",
            "input_head_sha": "",
            "input_skip_review": "false",
            "event_head_sha": "aaa111",
            "event_pr_number": "42",
            "event_clone_url": "https://github.com/org/repo.git",
            "lgtm_user": "LLLLKKKK",
            "output_file": "/dev/null",
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    @patch("ci_gate.review.github_get_pages")
    @patch("ci_gate.review.github_get")
    def test_no_review_returns_1(self, mock_get, mock_pages):
        """synchronize with no fresh review → exit 1 (FAILED check)."""
        mock_get.return_value = {
            "user": {"login": "author"},
            "head": {"sha": "aaa111", "repo": {"clone_url": "url"}},
            "state": "open",
        }
        mock_pages.return_value = []
        result = resolve_context(self._base_args())
        self.assertEqual(result, 1)

    @patch("ci_gate.review.github_get_pages")
    @patch("ci_gate.review.github_get")
    def test_approved_review_returns_0(self, mock_get, mock_pages):
        """synchronize with a fresh APPROVED review → exit 0 (SUCCESS check)."""
        mock_get.return_value = {
            "user": {"login": "author"},
            "head": {"sha": "aaa111", "repo": {"clone_url": "url"}},
            "state": "open",
        }
        mock_pages.return_value = [
            {"id": 1, "user": {"login": "rev", "type": "User"},
             "state": "APPROVED", "commit_id": "aaa111", "body": ""}
        ]
        result = resolve_context(self._base_args())
        self.assertEqual(result, 0)

    @patch("ci_gate.review.github_get")
    def test_head_changed_returns_1(self, mock_get):
        """synchronize but HEAD changed since event → exit 1."""
        mock_get.return_value = {
            "user": {"login": "author"},
            "head": {"sha": "bbb222", "repo": {"clone_url": "url"}},
            "state": "open",
        }
        result = resolve_context(self._base_args(event_head_sha="aaa111"))
        self.assertEqual(result, 1)

    @patch("ci_gate.review.github_get")
    def test_dispatch_closed_pr_returns_1(self, mock_get):
        """workflow_dispatch for a closed PR → exit 1."""
        mock_get.return_value = {
            "user": {"login": "author"},
            "head": {"sha": "aaa111", "repo": {"clone_url": "url"}},
            "state": "closed",
        }
        result = resolve_context(self._base_args(
            event_name="workflow_dispatch",
            input_pr_number="42",
            input_head_sha="aaa111",
        ))
        self.assertEqual(result, 1)

    @patch("ci_gate.review.github_get_pages")
    @patch("ci_gate.review.github_get")
    def test_dispatch_skip_review_returns_0(self, mock_get, mock_pages):
        """workflow_dispatch with skip_review_check=true → exit 0."""
        mock_get.return_value = {
            "user": {"login": "author"},
            "head": {"sha": "aaa111", "repo": {"clone_url": "url"}},
            "state": "open",
        }
        result = resolve_context(self._base_args(
            event_name="workflow_dispatch",
            input_pr_number="42",
            input_head_sha="aaa111",
            input_skip_review="true",
        ))
        self.assertEqual(result, 0)


# ---------------------------------------------------------------------------
# ci.pre_check_status (mocked)
# ---------------------------------------------------------------------------
class TestPreCheckStatus(unittest.TestCase):
    def _args(self, **overrides):
        defaults = {
            "commit_id": "abc123",
            "security": "secret",
            "repository": "repo",
            "max_attempts": 2,
            "sleep_interval": 0,
            "output_file": "",
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def _run_with_output(self, mock_status, status_response):
        mock_status.return_value = status_response
        with tempfile.NamedTemporaryFile(mode="r+") as output:
            result = pre_check_status(self._args(output_file=output.name))
            output.seek(0)
            return result, output.read()

    @patch("ci_gate.ci.retrieve_task_status")
    def test_done_returns_0(self, mock_status):
        result, output = self._run_with_output(mock_status, {"status": "SUCCESS", "commitId": "abc", "taskId": "1"})
        self.assertEqual(result, 0)
        self.assertIn("ci_action=done", output)

    @patch("ci_gate.ci.retrieve_task_status")
    def test_failed_returns_1(self, mock_status):
        result, output = self._run_with_output(mock_status, {"status": "FAILED", "commitId": "abc", "taskId": "1"})
        self.assertEqual(result, 1)
        self.assertIn("ci_action=trigger", output)

    @patch("ci_gate.ci.retrieve_task_status")
    def test_network_error_retry(self, mock_status):
        mock_status.side_effect = [
            GateError("Network error"),
            {"status": "SUCCESS", "commitId": "abc", "taskId": "1"},
        ]
        with tempfile.NamedTemporaryFile(mode="r+") as output:
            result = pre_check_status(self._args(output_file=output.name))
            output.seek(0)
            self.assertEqual(result, 0)
            self.assertIn("ci_action=done", output.read())
            self.assertEqual(mock_status.call_count, 2)

    @patch("ci_gate.ci.time.sleep")
    @patch("ci_gate.ci.retrieve_task_status")
    def test_running_returns_wait_action(self, mock_status, mock_sleep):
        mock_status.return_value = {"status": "RUNNING", "commitId": "abc", "taskId": "1"}
        with tempfile.NamedTemporaryFile(mode="r+") as output:
            result = pre_check_status(self._args(output_file=output.name))
            output.seek(0)
            self.assertEqual(result, 0)
            self.assertIn("ci_action=wait", output.read())

    @patch("ci_gate.ci.time.sleep")
    @patch("ci_gate.ci.retrieve_task_status")
    def test_pending_returns_trigger_action(self, mock_status, mock_sleep):
        mock_status.return_value = {"status": "PENDING", "commitId": "abc", "taskId": "1"}
        with tempfile.NamedTemporaryFile(mode="r+") as output:
            result = pre_check_status(self._args(output_file=output.name))
            output.seek(0)
            self.assertEqual(result, 1)
            self.assertIn("ci_action=trigger", output.read())


# ---------------------------------------------------------------------------
# ci.wait_status (mocked)
# ---------------------------------------------------------------------------
class TestWaitStatus(unittest.TestCase):
    def _args(self, **overrides):
        defaults = {
            "commit_id": "abc123",
            "security": "secret",
            "repository": "repo",
            "max_wait_time": 9999,
            "max_wait_pending_time": 9999,
            "max_wait_running_time": 9999,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    @patch("ci_gate.ci.time.sleep")
    @patch("ci_gate.ci.retrieve_task_status")
    def test_pending_then_done(self, mock_status, mock_sleep):
        mock_status.side_effect = [
            {"status": "PENDING", "commitId": "abc", "taskId": "1"},
            {"status": "RUNNING", "commitId": "abc", "taskId": "1"},
            {"status": "SUCCESS", "commitId": "abc", "taskId": "1"},
        ]
        result = wait_status(self._args())
        self.assertEqual(result, 0)

    @patch("ci_gate.ci.time.sleep")
    @patch("ci_gate.ci.retrieve_task_status")
    def test_failed_exits_early(self, mock_status, mock_sleep):
        mock_status.return_value = {"status": "FAILED", "commitId": "abc", "taskId": "1"}
        result = wait_status(self._args())
        self.assertEqual(result, 1)
        self.assertEqual(mock_status.call_count, 1)


# ---------------------------------------------------------------------------
# ci.trigger_ci (mocked)
# ---------------------------------------------------------------------------
class TestTriggerCi(unittest.TestCase):
    def _args(self, **overrides):
        defaults = {
            "commit_id": "abc123",
            "security": "secret",
            "github_source_repo": "https://github.com/org/repo.git",
            "github_pr_id": "42",
            "github_run_id": "100",
            "repository": "org/repo",
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    @patch("ci_gate.ci.ci_service_request")
    @patch("ci_gate.ci.get_branch_info")
    def test_branch_info_failure_raises(self, mock_branch, mock_ci):
        mock_branch.side_effect = GateError("Network error")
        with self.assertRaises(GateError):
            trigger_ci(self._args())

    @patch("ci_gate.ci.ci_service_request")
    @patch("ci_gate.ci.get_branch_info")
    def test_non_dict_response_raises(self, mock_branch, mock_ci):
        mock_branch.return_value = {"commit": {"id": "internal123"}}
        mock_ci.return_value = "OK"
        with self.assertRaises(GateError):
            trigger_ci(self._args())

    @patch("ci_gate.ci.ci_service_request")
    @patch("ci_gate.ci.get_branch_info")
    def test_success_response(self, mock_branch, mock_ci):
        mock_branch.return_value = {"commit": {"id": "internal123"}}
        mock_ci.return_value = {"success": True, "status": "CREATED"}
        result = trigger_ci(self._args())
        self.assertEqual(result, 0)

    @patch("ci_gate.ci.ci_service_request")
    @patch("ci_gate.ci.get_branch_info")
    def test_unknown_commit_raises(self, mock_branch, mock_ci):
        mock_branch.return_value = {"commit": {"id": None}}
        with self.assertRaises(GateError):
            trigger_ci(self._args())


# ---------------------------------------------------------------------------
# merge.trigger_merge (mocked)
# ---------------------------------------------------------------------------
class TestTriggerMerge(unittest.TestCase):
    def _args(self, **overrides):
        defaults = {
            "commit_id": "abc123",
            "security": "secret",
            "repository": "org/repo",
            "pr_id": "42",
            "author_email": "user@example.com",
            "author_name": "User",
            "merge_message": "merge commit",
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    @patch("ci_gate.merge.ci_service_request")
    def test_non_dict_response_raises(self, mock_ci):
        mock_ci.return_value = "OK"
        with self.assertRaises(GateError):
            trigger_merge(self._args())

    @patch("ci_gate.merge.ci_service_request")
    def test_success_response(self, mock_ci):
        mock_ci.return_value = {"success": True, "status": "CREATED"}
        result = trigger_merge(self._args())
        self.assertEqual(result, 0)


# ---------------------------------------------------------------------------
# merge.check_merge_conflicts (mocked)
# ---------------------------------------------------------------------------
class TestCheckMergeConflicts(unittest.TestCase):
    def _args(self, **overrides):
        defaults = {
            "pr_number": "42",
            "commit_id": "abc123",
            "repository": "repo",
            "github_token": "token",
            "max_retries": 2,
            "retry_interval": 0,
            "strict_mergeable": False,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    @patch("ci_gate.merge.github_get")
    def test_mergeable_true(self, mock_get):
        mock_get.side_effect = [
            {"mergeable": True},
            {"ahead_by": 0},
        ]
        result = check_merge_conflicts(self._args())
        self.assertEqual(result, 0)

    @patch("ci_gate.merge.github_get")
    def test_mergeable_false(self, mock_get):
        mock_get.return_value = {"mergeable": False}
        result = check_merge_conflicts(self._args())
        self.assertEqual(result, 1)

    @patch("ci_gate.merge.time.sleep")
    @patch("ci_gate.merge.github_get")
    def test_mergeable_null_retry_then_strict(self, mock_get, mock_sleep):
        mock_get.return_value = {"mergeable": None}
        result = check_merge_conflicts(self._args(strict_mergeable=True))
        self.assertEqual(result, 1)

    @patch("ci_gate.merge.time.sleep")
    @patch("ci_gate.merge.github_get")
    def test_mergeable_null_non_strict_proceeds(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            {"mergeable": None},
            {"mergeable": None},
            {"ahead_by": 3},
        ]
        result = check_merge_conflicts(self._args())
        self.assertEqual(result, 0)


# ---------------------------------------------------------------------------
# merge.wait_merge (mocked)
# ---------------------------------------------------------------------------
class TestWaitMerge(unittest.TestCase):
    def _args(self, **overrides):
        defaults = {
            "commit_id": "abc123",
            "security": "secret",
            "repository": "repo",
            "max_wait_time": 9999,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    @patch("ci_gate.merge.time.sleep")
    @patch("ci_gate.merge.ci_service_request")
    def test_pending_then_success(self, mock_request, mock_sleep):
        mock_request.side_effect = [
            {"status": "PENDING"},
            {"status": {"success": True}},
        ]
        result = wait_merge(self._args())
        self.assertEqual(result, 0)

    @patch("ci_gate.merge.time.sleep")
    @patch("ci_gate.merge.ci_service_request")
    def test_failure_raises(self, mock_request, mock_sleep):
        mock_request.return_value = {"status": {"success": False}}
        with self.assertRaises(GateError):
            wait_merge(self._args())


if __name__ == "__main__":
    unittest.main()
