"""Durable, per-request diagnostics; independent of strict acceptance receipts.

This module needs only the standard library. It never sends a model request,
retries a request, changes a sampling parameter, or imports a GPU runtime.
"""
from __future__ import annotations

import asyncio
import base64
from contextvars import ContextVar
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import uuid


_CALL = ContextVar("rtp_native_rpc_journal_call", default=None)


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _error(error):
    if error is None:
        return None
    value = dict(type=type(error).__name__, message=str(error))
    try:
        code = error.code()
        value["rpc_code"] = getattr(code, "name", str(code))
    except (AttributeError, TypeError):
        pass
    return value


class NativeRpcJournal:
    def __init__(self, directory, replica_rank, state, *, fsync=True):
        self.replica_rank = int(replica_rank)
        self.state = state
        self.fsync = bool(fsync)
        self.path = Path(directory) / f"rtp_rpc_requests_replica{self.replica_rank}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, record):
        data = (json.dumps(record, sort_keys=True, separators=(",", ":"),
                           allow_nan=False) + "\n").encode()
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            view = memoryview(data)
            while view:
                count = os.write(fd, view)
                if count <= 0:
                    raise OSError("request journal append made no progress")
                view = view[count:]
            if self.fsync:
                os.fsync(fd)
        finally:
            os.close(fd)

    def queued(self, request_id, ids, config, timeout, context, limit):
        tokens = list(ids)
        sampling = {name: getattr(config, name) for name in (
            "max_new_tokens", "temperature", "top_p", "top_k", "do_sample",
            "min_new_tokens", "ignore_eos")}
        try:
            sampling["seed"] = config.random_seed.value if config.HasField("random_seed") else None
        except (AttributeError, ValueError):
            sampling["seed"] = None
        # Preserve the configuration at _request entry. TokenRpcClient sets
        # timeout_ms/aux_info later; effective transport timeout is explicit.
        raw_config = None
        if hasattr(config, "SerializeToString"):
            raw_config = base64.b64encode(config.SerializeToString(deterministic=True)).decode()
        record = dict(schema="rtp-native-request-journal-v1", event="queued",
                      journal_id=uuid.uuid4().hex, call_id=context["call_id"],
                      kind=context["kind"], native_request_id=str(request_id),
                      replica_rank=self.replica_rank,
                      expected_model_version=context["expected_version"],
                      queued_unix=time.time(), queued_monotonic=time.monotonic(),
                      prompt_token_ids=tokens, prompt_tokens=len(tokens), sampling=sampling,
                      transport_timeout_seconds=float(timeout),
                      effective_transport_timeout_ms=max(1, int(timeout * 1000)),
                      max_active_requests=limit,
                      config_at_request_entry_pb_b64=raw_config,
                      **self.state())
        # Match the established strict-generation prompt SHA encoding exactly.
        record["prompt_sha256"] = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
        record["request_sha256"] = _sha({name: record[name] for name in (
            "native_request_id", "replica_rank", "kind", "expected_model_version",
            "prompt_token_ids", "sampling", "config_at_request_entry_pb_b64",
            "effective_transport_timeout_ms")})
        self._append(record)
        return dict(queued=record, start=None, transport_outcome="not_started", response=None)

    def start(self, ticket):
        record = dict(ticket["queued"], event="start", started_unix=time.time(),
                      started_monotonic=time.monotonic())
        record["queue_wait_seconds"] = record["started_monotonic"] - record["queued_monotonic"]
        self._append(record)  # Durable before the real native RPC is invoked.
        ticket["start"] = record
        ticket["transport_outcome"] = "not_returned"

    def finish(self, ticket, outcome, error):
        queued = ticket["queued"]
        start = ticket["start"]
        now = time.monotonic()
        record = {name: queued[name] for name in (
            "schema", "journal_id", "call_id", "kind", "native_request_id",
            "replica_rank", "expected_model_version", "request_sha256", "prompt_sha256")}
        record.update(event="finish", outcome=outcome, error=_error(error),
                      finished_unix=time.time(), finished_monotonic=now,
                      queued_unix=queued["queued_unix"], queued_monotonic=queued["queued_monotonic"],
                      started_unix=None if start is None else start["started_unix"],
                      started_monotonic=None if start is None else start["started_monotonic"],
                      queue_wait_seconds=(now - queued["queued_monotonic"] if start is None
                                          else start["queue_wait_seconds"]),
                      elapsed_seconds=now - queued["queued_monotonic"],
                      transport_outcome=ticket["transport_outcome"],
                      transport_finished_unix=ticket.get("transport_finished_unix"),
                      response=ticket["response"])
        self._append(record)


class JournaledTokenRpcMixin:
    """Place before the existing strict client in the MRO.

    The start record surrounds the real _request entry. Finish is emitted from
    generate/score_tokens finally, covering post-RPC token/version validation.
    ContextVar state remains separate for concurrent calls on one client.
    """
    def __init__(self, *args, request_journal, max_active_requests=None, **kwargs):
        self._request_journal = request_journal
        limit = int(os.environ.get("RLLM_RTP_MAX_ACTIVE_REQUESTS", "0")) if max_active_requests is None else int(max_active_requests)
        if limit < 0:
            raise ValueError("RLLM_RTP_MAX_ACTIVE_REQUESTS must be nonnegative")
        self._max_active_requests = limit
        self._request_gate = asyncio.Semaphore(limit) if limit else None
        self._admission_open = True
        self._admission_epoch = 0
        self._admission_reason = None
        super().__init__(*args, **kwargs)

    def close_admission(self, reason):
        """Close new/queued leases without cancelling an already active RPC."""
        self._admission_open = False
        self._admission_epoch += 1
        self._admission_reason = str(reason)

    def open_admission(self):
        self._admission_open = True
        self._admission_reason = None

    def _check_admission(self, epoch):
        if not self._admission_open or epoch != self._admission_epoch:
            raise NativeRpcAdmissionClosed("RPC lease retired or admission closed: " +
                                           str(self._admission_reason))

    async def _journal_call(self, kind, version, invoke, args, kwargs):
        context = dict(call_id=uuid.uuid4().hex, kind=kind, expected_version=version, tickets=[])
        token = _CALL.set(context)
        outcome, error = "error", None
        try:
            result = await invoke(*args, expected_version=version, **kwargs)
            outcome = "success"
            return result
        except asyncio.CancelledError as caught:
            outcome, error = "cancelled", caught
            raise
        except BaseException as caught:
            error = caught
            raise
        finally:
            try:
                for ticket in context["tickets"]:
                    self._request_journal.finish(ticket, outcome, error)
            except Exception as journal_error:
                if error is None:
                    raise
                # Preserve the real RPC/cancellation error if diagnostic I/O
                # also failed. Do not silently retry either operation.
                if hasattr(error, "add_note"):
                    error.add_note("request finish journal failed: " + repr(journal_error))
                else:
                    import logging
                    logging.getLogger(__name__).exception("request finish journal failed")
            finally:
                for ticket in context["tickets"]:
                    if ticket.get("lease_acquired"):
                        ticket["lease_acquired"] = False
                        self._request_gate.release()
                _CALL.reset(token)

    async def generate(self, *args, expected_version=None, **kwargs):
        return await self._journal_call("generate", expected_version, super().generate, args, kwargs)

    async def score_tokens(self, *args, expected_version=None, **kwargs):
        return await self._journal_call("score_tokens", expected_version, super().score_tokens, args, kwargs)

    async def close(self):
        self.close_admission("client_close")
        return await super().close()

    async def _request(self, request_id, ids, config):
        context = _CALL.get()
        if context is None:
            raise RuntimeError("journaled native _request requires generate/score_tokens context")
        ticket = self._request_journal.queued(request_id, ids, config, self.timeout,
                                              context, self._max_active_requests)
        context["tickets"].append(ticket)
        epoch = self._admission_epoch
        try:
            self._check_admission(epoch)
            if self._request_gate is not None:
                await self._request_gate.acquire()
                ticket["lease_acquired"] = True
            # No await between this second check and actual RPC entry. Old
            # queued requests stay rejected even if a later probe reopens.
            self._check_admission(epoch)
            self._request_journal.start(ticket)
            response = await super()._request(request_id, ids, config)
            ticket["transport_outcome"] = "response_received"
            ticket["response"] = dict(model_version=getattr(response, "model_version", None),
                                      global_step=getattr(response, "global_step", None))
            return response
        except asyncio.CancelledError:
            ticket["transport_outcome"] = "cancelled"
            raise
        except BaseException:
            ticket["transport_outcome"] = "error"
            raise
        finally:
            ticket["transport_finished_unix"] = time.time()
            # The enclosing generate/score_tokens finally writes its validated
            # terminal record and releases the one lease, also on cancellation.


class NativeRpcAdmissionClosed(RuntimeError):
    pass
