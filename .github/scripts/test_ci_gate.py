"""Unit tests for ci_gate package (mock HTTP, no network calls)."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from ci_gate.common import GateError, _lower_headers, is_true
from ci_gate.ci_service import collect_status_tokens, parse_ci_status
from ci_gate.review import (
    _check_issue_comments_qualified,
    check_review_qualified,
    latest_fresh_reviews,
    resolve_context,
)
from ci_gate.ci import pre_check_status, trigger_ci, wait_status
from ci_gate.merge import check_merge_conflicts, trigger_merge, wait_merge
from ci_gate.github import is_rate_limited
from ci_gate.rerun import DEDUPE_MARKER, rerun_pr_build


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
    def _review(self, login, state, commit_id="abc123", user_type="User", review_id=1, body="",
                association="MEMBER"):
        return {
            "id": review_id,
            "user": {"login": login, "type": user_type},
            "state": state,
            "commit_id": commit_id,
            "body": body,
            "author_association": association,
        }

    def test_filters_wrong_commit(self):
        reviews = [self._review("alice", "APPROVED", commit_id="other")]
        result = latest_fresh_reviews(reviews, "abc123", "author")
        self.assertEqual(result, [])

    def test_filters_bot(self):
        reviews = [self._review("bot", "APPROVED", user_type="Bot")]
        result = latest_fresh_reviews(reviews, "abc123", "author")
        self.assertEqual(result, [])

    def test_filters_untrusted_association(self):
        reviews = [
            self._review("stranger", "APPROVED", association="NONE"),
            self._review("drive-by", "CHANGES_REQUESTED", association="FIRST_TIME_CONTRIBUTOR", review_id=2),
            self._review("forker", "APPROVED", association="CONTRIBUTOR", review_id=3),
            self._review("no-assoc", "APPROVED", association="", review_id=4),
        ]
        result = latest_fresh_reviews(reviews, "abc123", "author")
        self.assertEqual(result, [])

    def test_keeps_trusted_associations(self):
        reviews = [
            self._review("owner", "APPROVED", association="OWNER", review_id=1),
            self._review("member", "APPROVED", association="MEMBER", review_id=2),
            self._review("collab", "APPROVED", association="COLLABORATOR", review_id=3),
        ]
        result = latest_fresh_reviews(reviews, "abc123", "author")
        self.assertEqual(len(result), 3)

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
             "state": "APPROVED", "commit_id": "sha1", "body": "", "author_association": "MEMBER"}
        ]
        result = check_review_qualified("1", "repo", "sha1", "token", "LLLLKKKK")
        self.assertTrue(result)

    @patch("ci_gate.review.github_get_pages")
    @patch("ci_gate.review.github_get")
    def test_changes_requested(self, mock_get, mock_pages):
        mock_get.return_value = self._mock_pr()
        mock_pages.return_value = [
            {"id": 1, "user": {"login": "reviewer", "type": "User"},
             "state": "CHANGES_REQUESTED", "commit_id": "sha1", "body": "", "author_association": "MEMBER"}
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
             "body": "lgtm ready to ci please", "author_association": "MEMBER"}
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

    @patch("ci_gate.review.github_get_pages")
    @patch("ci_gate.review.github_get")
    def test_no_run_id_skips_comment_path(self, mock_get, mock_pages):
        """Without a run id there is no trustworthy anchor, so comments are skipped."""
        mock_get.return_value = self._mock_pr()
        mock_pages.return_value = []
        self.assertFalse(check_review_qualified("1", "repo", "sha1", "token", "LLLLKKKK"))
        paths = [call.args[1] for call in mock_pages.call_args_list]
        self.assertEqual(paths, ["/pulls/1/reviews"])


# ---------------------------------------------------------------------------
# review._check_issue_comments_qualified (mocked)
# ---------------------------------------------------------------------------
class TestCheckIssueCommentsQualified(unittest.TestCase):
    PUSHED_AT = "2025-04-20T10:00:00Z"
    RUN_RESPONSE = {"created_at": PUSHED_AT}

    def _comment(self, login="LLLLKKKK", body="lgtm ready to ci", updated_at="2025-04-20T12:00:00Z"):
        return {"user": {"login": login}, "body": body, "updated_at": updated_at}

    def _check(self, comments, lgtm_user="LLLLKKKK", pr_author="", run_id="42",
               run_response=None):
        with patch("ci_gate.review.github_get",
                   return_value=self.RUN_RESPONSE if run_response is None else run_response), \
             patch("ci_gate.review.github_get_pages", return_value=comments):
            return _check_issue_comments_qualified(
                "1", "repo", "token", lgtm_user, pr_author, run_id)

    def test_comment_after_push_qualifies(self):
        self.assertTrue(self._check([self._comment()]))

    def test_comment_before_push_rejected(self):
        self.assertFalse(self._check([self._comment(updated_at="2025-04-20T09:00:00Z")]))

    def test_backdated_commit_cannot_revive_stale_comment(self):
        """The anchor is the run creation time, so commit dates cannot shift it."""
        stale = self._comment(updated_at="2025-04-20T09:59:59Z")
        self.assertFalse(self._check([stale]))

    def test_wrong_author_rejected(self):
        self.assertFalse(self._check([self._comment(login="other-user")]))

    def test_missing_phrase_rejected(self):
        self.assertFalse(self._check([self._comment(body="looks good")]))

    def test_no_comments(self):
        self.assertFalse(self._check([]))

    def test_case_insensitive_phrase(self):
        self.assertTrue(self._check([self._comment(body="LGTM Ready To CI")]))

    def test_picks_latest_matching_comment(self):
        comments = [
            self._comment(updated_at="2025-04-20T11:00:00Z"),
            self._comment(updated_at="2025-04-20T14:00:00Z"),
        ]
        self.assertTrue(self._check(comments))

    def test_multi_lgtm_users_second_user_qualifies(self):
        self.assertTrue(self._check([self._comment(login="netaddi")], lgtm_user="LLLLKKKK,netaddi"))

    def test_pr_author_self_lgtm_rejected(self):
        self.assertFalse(self._check(
            [self._comment(login="netaddi")], lgtm_user="LLLLKKKK,netaddi", pr_author="netaddi"))

    def test_missing_run_id_rejected(self):
        self.assertFalse(self._check([self._comment()], run_id=""))

    def test_run_without_created_at_rejected(self):
        self.assertFalse(self._check([self._comment()], run_response={}))



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
            "run_id": "",
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
        mock_qualified.assert_called_once_with("42", "org/repo", "abc111", "tok", "LLLLKKKK", "")

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
            "run_id": "",
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
             "state": "APPROVED", "commit_id": "aaa111", "body": "", "author_association": "MEMBER"}
        ]
        result = resolve_context(self._base_args())
        self.assertEqual(result, 0)

    @patch("ci_gate.review.check_review_qualified")
    @patch("ci_gate.review.github_get")
    def test_run_id_forwarded_for_pull_request(self, mock_get, mock_qualified):
        """pull_request runs are created at push time, so their id anchors comments."""
        mock_get.return_value = {
            "user": {"login": "author"},
            "head": {"sha": "aaa111", "repo": {"clone_url": "url"}},
            "state": "open",
        }
        mock_qualified.return_value = True
        resolve_context(self._base_args(run_id="999"))
        self.assertEqual(mock_qualified.call_args.args[-1], "999")

    @patch("ci_gate.review.check_review_qualified")
    @patch("ci_gate.review.github_get")
    def test_run_id_dropped_for_workflow_dispatch(self, mock_get, mock_qualified):
        """A dispatch run starts now, so it must not anchor LGTM comments."""
        mock_get.return_value = {
            "user": {"login": "author"},
            "head": {"sha": "aaa111", "repo": {"clone_url": "url"}},
            "state": "open",
        }
        mock_qualified.return_value = True
        resolve_context(self._base_args(
            event_name="workflow_dispatch", input_pr_number="42",
            input_head_sha="aaa111", run_id="999"))
        self.assertEqual(mock_qualified.call_args.args[-1], "")

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


class TestIsRateLimited(unittest.TestCase):
    """A permission denial must never be retried as throttling."""

    def test_permission_denial_is_not_rate_limiting(self):
        body = {"message": "Resource not accessible by integration"}
        self.assertFalse(is_rate_limited(403, {}, body))

    def test_permission_denial_wins_over_exhausted_quota_header(self):
        """The short-circuit has to come first, or the fork bug comes back."""
        body = {"message": "Resource not accessible by integration"}
        headers = {"x-ratelimit-remaining": "0"}
        self.assertFalse(is_rate_limited(403, headers, body))

    def test_429_is_rate_limiting(self):
        self.assertTrue(is_rate_limited(429, {}, {"message": "slow down"}))

    def test_exhausted_quota_header_is_rate_limiting(self):
        self.assertTrue(is_rate_limited(403, {"x-ratelimit-remaining": "0"}, {}))

    def test_retry_after_header_is_rate_limiting(self):
        self.assertTrue(is_rate_limited(403, {"retry-after": "60"}, {}))

    def test_rate_limit_prose_is_rate_limiting(self):
        body = {"message": "API rate limit exceeded for installation"}
        self.assertTrue(is_rate_limited(403, {}, body))

    def test_secondary_and_content_limits_are_rate_limiting(self):
        self.assertTrue(is_rate_limited(
            403, {}, {"message": "You have exceeded a secondary rate limit"}))
        self.assertTrue(is_rate_limited(
            403, {}, {"message": "were submitted too quickly"}))

    def test_reset_header_alone_is_not_rate_limiting(self):
        """x-ratelimit-reset rides on every response and is always in future."""
        headers = {"x-ratelimit-reset": "99999999999",
                   "x-ratelimit-remaining": "4999"}
        self.assertFalse(is_rate_limited(403, headers, {"message": "Forbidden"}))

    def test_none_body_does_not_raise(self):
        self.assertFalse(is_rate_limited(403, {}, None))
        self.assertTrue(is_rate_limited(429, {}, None))

    def test_other_statuses_are_not_rate_limiting(self):
        self.assertFalse(is_rate_limited(500, {}, {"message": "boom"}))

    def test_abuse_detection_is_rate_limiting(self):
        """Carries neither "rate limit" nor "submitted too quickly"."""
        body = {"message": "You have triggered an abuse detection mechanism. "
                           "Please retry your request again later."}
        self.assertTrue(is_rate_limited(403, {}, body))

    def test_fine_grained_token_denial_is_permanent(self):
        """Recommended fix #1 in the workflow header may switch token types."""
        body = {"message": "Resource not accessible by personal access token"}
        self.assertFalse(is_rate_limited(403, {}, body))

    def test_zero_retry_after_is_not_rate_limiting(self):
        self.assertFalse(is_rate_limited(403, {"retry-after": "0"}, {"message": "no"}))


class TestLowerHeaders(unittest.TestCase):
    """Header names are case-insensitive on the wire; dict lookups are not."""

    def test_server_casing_is_normalized(self):
        headers = _lower_headers({"X-RateLimit-Remaining": "0",
                                  "Retry-After": "60"})
        self.assertEqual(headers["x-ratelimit-remaining"], "0")
        self.assertEqual(headers["retry-after"], "60")

    def test_normalized_headers_reach_the_classifier(self):
        """Without normalization every lookup would silently miss."""
        headers = _lower_headers({"X-RateLimit-Remaining": "0"})
        self.assertTrue(is_rate_limited(403, headers, {"message": "Forbidden"}))

    def test_empty_and_none_are_safe(self):
        self.assertEqual(_lower_headers(None), {})
        self.assertEqual(_lower_headers({}), {})

    def test_email_message_headers_are_supported(self):
        """The real source is email.message.Message from urllib."""
        import email.message

        message = email.message.Message()
        message["X-RateLimit-Remaining"] = "0"
        self.assertEqual(_lower_headers(message)["x-ratelimit-remaining"], "0")


class TestRerunPrBuild(unittest.TestCase):
    """rerun.py had no coverage at all; these pin old and new behaviour."""

    def _args(self, **overrides):
        defaults = {
            "repository": "alibaba/rtp-llm",
            "pr_number": "1285",
            "head_sha": "42941eb7" + "0" * 32,
            "workflow_file": "CI-request-trigger.yml",
            "github_token": "token",
            "max_retries": 3,
            "retry_backoff": 2.0,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    @staticmethod
    def _run(**overrides):
        run = {
            "id": 31458188287,
            "status": "completed",
            "conclusion": "failure",
            "head_repository": {"full_name": "alibaba/rtp-llm"},
        }
        run.update(overrides)
        return run

    # ---------------- baseline: pre-existing branches ----------------

    @patch("ci_gate.rerun.post_pr_comment")
    @patch("ci_gate.rerun.list_workflow_runs")
    def test_no_run_posts_comment_and_returns_0(self, mock_runs, mock_comment):
        mock_runs.return_value = []
        mock_comment.return_value = (201, {}, {})
        self.assertEqual(rerun_pr_build(self._args()), 0)
        self.assertEqual(mock_comment.call_count, 1)

    @patch("ci_gate.rerun.rerun_workflow_run")
    @patch("ci_gate.rerun.list_workflow_runs")
    def test_active_run_is_left_alone(self, mock_runs, mock_rerun):
        mock_runs.return_value = [self._run(status="in_progress", conclusion=None)]
        self.assertEqual(rerun_pr_build(self._args()), 0)
        mock_rerun.assert_not_called()

    @patch("ci_gate.rerun.rerun_workflow_run")
    @patch("ci_gate.rerun.list_workflow_runs")
    def test_successful_run_is_left_alone(self, mock_runs, mock_rerun):
        mock_runs.return_value = [self._run(conclusion="success")]
        self.assertEqual(rerun_pr_build(self._args()), 0)
        mock_rerun.assert_not_called()

    @patch("ci_gate.rerun.rerun_workflow_run")
    @patch("ci_gate.rerun.list_workflow_runs")
    def test_rerun_triggered_returns_0(self, mock_runs, mock_rerun):
        mock_runs.return_value = [self._run()]
        mock_rerun.return_value = (201, {}, {})
        self.assertEqual(rerun_pr_build(self._args()), 0)
        self.assertEqual(mock_rerun.call_count, 1)

    @patch("ci_gate.rerun.rerun_workflow_run")
    @patch("ci_gate.rerun.list_workflow_runs")
    def test_conflict_409_returns_0(self, mock_runs, mock_rerun):
        mock_runs.return_value = [self._run()]
        mock_rerun.return_value = (409, {"message": "already running"}, {})
        self.assertEqual(rerun_pr_build(self._args()), 0)

    @patch("ci_gate.rerun.post_pr_comment")
    @patch("ci_gate.rerun.rerun_workflow_run")
    @patch("ci_gate.rerun.list_workflow_runs")
    def test_expired_422_posts_comment_and_returns_0(self, mock_runs, mock_rerun,
                                                     mock_comment):
        mock_runs.return_value = [self._run()]
        mock_rerun.return_value = (422, {"message": "too old"}, {})
        mock_comment.return_value = (201, {}, {})
        self.assertEqual(rerun_pr_build(self._args()), 0)
        self.assertEqual(mock_comment.call_count, 1)

    # ---------------- new: action_required needs approval, not rerun ----------

    @patch("ci_gate.rerun.github_get_pages")
    @patch("ci_gate.rerun.post_pr_comment")
    @patch("ci_gate.rerun.rerun_workflow_run")
    @patch("ci_gate.rerun.list_workflow_runs")
    def test_action_required_conclusion_skips_rerun(self, mock_runs, mock_rerun,
                                                    mock_comment, mock_pages):
        """It arrives as a conclusion with status=completed, not as a status."""
        mock_runs.return_value = [self._run(
            status="completed", conclusion="action_required",
            head_repository={"full_name": "Vinkle-hzt/rtp-llm"})]
        mock_pages.return_value = []
        mock_comment.return_value = (201, {}, {})

        self.assertEqual(rerun_pr_build(self._args()), 0)

        mock_rerun.assert_not_called()
        body = mock_comment.call_args[0][2]
        self.assertIn("Approve and run workflows", body)
        self.assertIn("Vinkle-hzt/rtp-llm", body)
        self.assertIn("Do NOT push an empty commit", body)

    # ---------------- new: permission 403 is not throttling ----------------

    @patch("ci_gate.rerun.github_get_pages")
    @patch("ci_gate.rerun.post_pr_comment")
    @patch("ci_gate.rerun.rerun_workflow_run")
    @patch("ci_gate.rerun.list_workflow_runs")
    def test_fork_permission_403_explains_and_returns_0(self, mock_runs, mock_rerun,
                                                        mock_comment, mock_pages):
        mock_runs.return_value = [self._run(
            head_repository={"full_name": "Vinkle-hzt/rtp-llm"})]
        mock_rerun.return_value = (
            403, {"message": "Resource not accessible by integration"}, {})
        mock_pages.return_value = []
        mock_comment.return_value = (201, {}, {})

        self.assertEqual(rerun_pr_build(self._args()), 0)

        self.assertEqual(mock_rerun.call_count, 1,
                         "a permission denial must not be retried")
        body = mock_comment.call_args[0][2]
        self.assertIn("belongs to a fork", body)
        self.assertIn("Vinkle-hzt/rtp-llm", body)
        self.assertIn("Do NOT push an empty commit", body)

    @patch("ci_gate.rerun.github_get_pages")
    @patch("ci_gate.rerun.post_pr_comment")
    @patch("ci_gate.rerun.rerun_workflow_run")
    @patch("ci_gate.rerun.list_workflow_runs")
    def test_same_repo_permission_403_keeps_reapprove_hint(self, mock_runs, mock_rerun,
                                                           mock_comment, mock_pages):
        mock_runs.return_value = [self._run()]
        mock_rerun.return_value = (
            403, {"message": "Resource not accessible by integration"}, {})
        mock_pages.return_value = []
        mock_comment.return_value = (201, {}, {})

        self.assertEqual(rerun_pr_build(self._args()), 0)

        body = mock_comment.call_args[0][2]
        self.assertNotIn("belongs to a fork", body)
        self.assertIn("lgtm ready to ci", body)

    @patch("ci_gate.rerun.github_get_pages")
    @patch("ci_gate.rerun.post_pr_comment")
    @patch("ci_gate.rerun.rerun_workflow_run")
    @patch("ci_gate.rerun.list_workflow_runs")
    def test_deleted_fork_head_repository_gets_fork_wording(self, mock_runs, mock_rerun,
                                                            mock_comment, mock_pages):
        """A null head_repository means the fork was deleted, not that it is ours."""
        mock_runs.return_value = [self._run(head_repository=None)]
        mock_rerun.return_value = (
            403, {"message": "Resource not accessible by integration"}, {})
        mock_pages.return_value = []
        mock_comment.return_value = (201, {}, {})

        self.assertEqual(rerun_pr_build(self._args()), 0)

        self.assertIn("belongs to a fork", mock_comment.call_args[0][2])

    @patch("ci_gate.rerun.github_get_pages")
    @patch("ci_gate.rerun.post_pr_comment")
    @patch("ci_gate.rerun.rerun_workflow_run")
    @patch("ci_gate.rerun.list_workflow_runs")
    def test_action_required_status_arm(self, mock_runs, mock_rerun,
                                       mock_comment, mock_pages):
        mock_runs.return_value = [self._run(
            status="action_required", conclusion=None,
            head_repository={"full_name": "Vinkle-hzt/rtp-llm"})]
        mock_pages.return_value = []
        mock_comment.return_value = (201, {}, {})
        self.assertEqual(rerun_pr_build(self._args()), 0)
        mock_rerun.assert_not_called()

    @patch("ci_gate.rerun.github_get_pages")
    @patch("ci_gate.rerun.post_pr_comment")
    @patch("ci_gate.rerun.rerun_workflow_run")
    @patch("ci_gate.rerun.list_workflow_runs")
    def test_same_repo_action_required_does_not_claim_a_fork(
            self, mock_runs, mock_rerun, mock_comment, mock_pages):
        """An environment gate can hold a same-repo run at action_required."""
        mock_runs.return_value = [self._run(conclusion="action_required")]
        mock_pages.return_value = []
        mock_comment.return_value = (201, {}, {})

        self.assertEqual(rerun_pr_build(self._args()), 0)

        body = mock_comment.call_args[0][2]
        self.assertNotIn("from a fork", body)
        self.assertIn("action_required", body)

    # ---------------- new: real throttling still retries ----------------

    @patch("ci_gate.rerun.time.sleep")
    @patch("ci_gate.rerun.rerun_workflow_run")
    @patch("ci_gate.rerun.list_workflow_runs")
    def test_real_rate_limit_retries_then_fails_hard(self, mock_runs, mock_rerun,
                                                     mock_sleep):
        mock_runs.return_value = [self._run()]
        mock_rerun.return_value = (
            403, {"message": "API rate limit exceeded"},
            {"x-ratelimit-remaining": "0"})
        with self.assertRaises(GateError) as ctx:
            rerun_pr_build(self._args())
        self.assertEqual(ctx.exception.exit_code, 2)
        self.assertEqual(mock_rerun.call_count, 3)

    @patch("ci_gate.rerun.time.sleep")
    @patch("ci_gate.rerun.rerun_workflow_run")
    @patch("ci_gate.rerun.list_workflow_runs")
    def test_rate_limit_then_success(self, mock_runs, mock_rerun, mock_sleep):
        mock_runs.return_value = [self._run()]
        mock_rerun.side_effect = [
            (429, {"message": "slow down"}, {}),
            (201, {}, {}),
        ]
        self.assertEqual(rerun_pr_build(self._args()), 0)
        self.assertEqual(mock_rerun.call_count, 2)

    # ---------------- new: comment dedupe ----------------

    @patch("ci_gate.rerun.github_get_pages")
    @patch("ci_gate.rerun.post_pr_comment")
    @patch("ci_gate.rerun.rerun_workflow_run")
    @patch("ci_gate.rerun.list_workflow_runs")
    def test_existing_marker_suppresses_duplicate_comment(self, mock_runs, mock_rerun,
                                                          mock_comment, mock_pages):
        args = self._args()
        mock_runs.return_value = [self._run(
            head_repository={"full_name": "Vinkle-hzt/rtp-llm"})]
        mock_rerun.return_value = (
            403, {"message": "Resource not accessible by integration"}, {})
        mock_pages.return_value = [
            {"body": DEDUPE_MARKER % (args.head_sha, "forbidden")
                     + "\n\nearlier explanation"},
        ]
        self.assertEqual(rerun_pr_build(args), 0)
        mock_comment.assert_not_called()

    @patch("ci_gate.rerun.github_get_pages")
    @patch("ci_gate.rerun.post_pr_comment")
    @patch("ci_gate.rerun.rerun_workflow_run")
    @patch("ci_gate.rerun.list_workflow_runs")
    def test_marker_ignores_run_id_so_a_new_run_does_not_repost(
            self, mock_runs, mock_rerun, mock_comment, mock_pages):
        """Same SHA and reason, new run id: a run-scoped key would have spammed."""
        args = self._args()
        mock_runs.return_value = [self._run(
            id=99999999999, head_repository={"full_name": "Vinkle-hzt/rtp-llm"})]
        mock_rerun.return_value = (
            403, {"message": "Resource not accessible by integration"}, {})
        mock_pages.return_value = [
            {"body": DEDUPE_MARKER % (args.head_sha, "forbidden")
                     + "\n\nfrom run 31458188287"},
        ]
        self.assertEqual(rerun_pr_build(args), 0)
        mock_comment.assert_not_called()

    @patch("ci_gate.rerun.github_get_pages")
    @patch("ci_gate.rerun.post_pr_comment")
    @patch("ci_gate.rerun.rerun_workflow_run")
    @patch("ci_gate.rerun.list_workflow_runs")
    def test_marker_for_another_sha_does_not_suppress(self, mock_runs, mock_rerun,
                                                      mock_comment, mock_pages):
        """The suppression half alone would pass with a constant marker."""
        mock_runs.return_value = [self._run(
            head_repository={"full_name": "Vinkle-hzt/rtp-llm"})]
        mock_rerun.return_value = (
            403, {"message": "Resource not accessible by integration"}, {})
        mock_pages.return_value = [
            {"body": DEDUPE_MARKER % ("f" * 40, "forbidden") + "\n\nold head"},
        ]
        mock_comment.return_value = (201, {}, {})
        self.assertEqual(rerun_pr_build(self._args()), 0)
        self.assertEqual(mock_comment.call_count, 1)

    @patch("ci_gate.rerun.github_get_pages")
    @patch("ci_gate.rerun.post_pr_comment")
    @patch("ci_gate.rerun.rerun_workflow_run")
    @patch("ci_gate.rerun.list_workflow_runs")
    def test_approval_marker_does_not_suppress_the_forbidden_comment(
            self, mock_runs, mock_rerun, mock_comment, mock_pages):
        """The real sequence: approval advice first, then a rerun denial.

        A SHA-only key would leave the PR showing stale advice to approve a run
        that has already been approved and run.
        """
        args = self._args()
        mock_runs.return_value = [self._run(
            head_repository={"full_name": "Vinkle-hzt/rtp-llm"})]
        mock_rerun.return_value = (
            403, {"message": "Resource not accessible by integration"}, {})
        mock_pages.return_value = [
            {"body": DEDUPE_MARKER % (args.head_sha, "approval")
                     + "\n\nclick Approve and run workflows"},
        ]
        mock_comment.return_value = (201, {}, {})

        self.assertEqual(rerun_pr_build(args), 0)

        self.assertEqual(mock_comment.call_count, 1)
        self.assertIn("belongs to a fork", mock_comment.call_args[0][2])

    @patch("ci_gate.rerun.github_get_pages")
    @patch("ci_gate.rerun.post_pr_comment")
    @patch("ci_gate.rerun.rerun_workflow_run")
    @patch("ci_gate.rerun.list_workflow_runs")
    def test_dedupe_read_failure_does_not_escalate_to_exit_2(
            self, mock_runs, mock_rerun, mock_comment, mock_pages):
        """github_get_pages raises GateError(2) on any non-200."""
        mock_runs.return_value = [self._run(
            head_repository={"full_name": "Vinkle-hzt/rtp-llm"})]
        mock_rerun.return_value = (
            403, {"message": "Resource not accessible by integration"}, {})
        mock_pages.side_effect = GateError("::error::boom", 2)
        mock_comment.return_value = (201, {}, {})

        self.assertEqual(rerun_pr_build(self._args()), 0)

        self.assertEqual(mock_comment.call_count, 1)

    @patch("ci_gate.rerun.github_get_pages")
    @patch("ci_gate.rerun.post_pr_comment")
    @patch("ci_gate.rerun.rerun_workflow_run")
    @patch("ci_gate.rerun.list_workflow_runs")
    def test_dedupe_read_raising_bare_oserror_does_not_escalate(
            self, mock_runs, mock_rerun, mock_comment, mock_pages):
        """urllib lets socket.timeout and BadStatusLine through unwrapped."""
        mock_runs.return_value = [self._run(
            head_repository={"full_name": "Vinkle-hzt/rtp-llm"})]
        mock_rerun.return_value = (
            403, {"message": "Resource not accessible by integration"}, {})
        mock_pages.side_effect = OSError("connection reset")
        mock_comment.return_value = (201, {}, {})
        self.assertEqual(rerun_pr_build(self._args()), 0)

    @patch("ci_gate.rerun.github_get_pages")
    @patch("ci_gate.rerun.post_pr_comment")
    @patch("ci_gate.rerun.rerun_workflow_run")
    @patch("ci_gate.rerun.list_workflow_runs")
    def test_comment_post_failure_warns_but_returns_0(self, mock_runs, mock_rerun,
                                                      mock_comment, mock_pages):
        mock_runs.return_value = [self._run(
            head_repository={"full_name": "Vinkle-hzt/rtp-llm"})]
        mock_rerun.return_value = (
            403, {"message": "Resource not accessible by integration"}, {})
        mock_pages.return_value = []
        mock_comment.return_value = (403, {"message": "denied"}, {})
        self.assertEqual(rerun_pr_build(self._args()), 0)

    @patch("ci_gate.rerun.github_get_pages")
    @patch("ci_gate.rerun.post_pr_comment")
    @patch("ci_gate.rerun.rerun_workflow_run")
    @patch("ci_gate.rerun.list_workflow_runs")
    def test_comment_post_raising_does_not_escalate(self, mock_runs, mock_rerun,
                                                    mock_comment, mock_pages):
        """One network blip while posting must not become exit 2."""
        mock_runs.return_value = [self._run(
            head_repository={"full_name": "Vinkle-hzt/rtp-llm"})]
        mock_rerun.return_value = (
            403, {"message": "Resource not accessible by integration"}, {})
        mock_pages.return_value = []
        mock_comment.side_effect = GateError("::error::Network error", 2)
        self.assertEqual(rerun_pr_build(self._args()), 0)

    # ---------------- unexpected statuses still fail hard ----------------

    @patch("ci_gate.rerun.rerun_workflow_run")
    @patch("ci_gate.rerun.list_workflow_runs")
    def test_unexpected_status_raises_exit_2(self, mock_runs, mock_rerun):
        mock_runs.return_value = [self._run()]
        mock_rerun.return_value = (500, {"message": "server error"}, {})
        with self.assertRaises(GateError) as ctx:
            rerun_pr_build(self._args())
        self.assertEqual(ctx.exception.exit_code, 2)
        self.assertEqual(mock_rerun.call_count, 1)


if __name__ == "__main__":
    unittest.main()
