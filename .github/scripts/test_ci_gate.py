"""Unit tests for ci_gate package (mock HTTP, no network calls)."""

from __future__ import annotations

import argparse
import json
import unittest
from unittest.mock import MagicMock, patch

from ci_gate.common import GateError, is_true
from ci_gate.ci_service import collect_status_tokens, parse_ci_status
from ci_gate.review import check_review_qualified, latest_fresh_reviews
from ci_gate.ci import pre_check_status, wait_status
from ci_gate.merge import check_merge_conflicts, wait_merge


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
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    @patch("ci_gate.ci.retrieve_task_status")
    def test_done_returns_0(self, mock_status):
        mock_status.return_value = {"status": "SUCCESS", "commitId": "abc", "taskId": "1"}
        result = pre_check_status(self._args())
        self.assertEqual(result, 0)

    @patch("ci_gate.ci.retrieve_task_status")
    def test_failed_returns_1(self, mock_status):
        mock_status.return_value = {"status": "FAILED", "commitId": "abc", "taskId": "1"}
        result = pre_check_status(self._args())
        self.assertEqual(result, 1)

    @patch("ci_gate.ci.retrieve_task_status")
    def test_network_error_retry(self, mock_status):
        mock_status.side_effect = [
            GateError("Network error"),
            {"status": "SUCCESS", "commitId": "abc", "taskId": "1"},
        ]
        result = pre_check_status(self._args())
        self.assertEqual(result, 0)
        self.assertEqual(mock_status.call_count, 2)


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
