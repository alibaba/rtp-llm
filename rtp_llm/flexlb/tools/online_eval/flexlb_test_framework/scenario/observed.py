"""Opt-in caller observations for the two legacy RPC latency contracts."""

from .backend import RequestBatch


class ObservedRequestBatch(RequestBatch):
    """One caller-opened RPC, using the ordinary consumer and ownership ledger.

    RPC lifetime, first-output observation, end observation and cleanup grace
    are separate budgets. Ordinary RequestBatch submit/wait stay unchanged.
    """

    def __init__(self, ctx, params, mode, deadline, end_wait_s=15):
        if params.get("count") != 1 or params.get("consume") != "immediate":
            raise ValueError("observed stream requires one immediate request")
        if mode not in ("stream_ttft", "request_total"):
            raise ValueError("unknown caller observation mode")
        if type(end_wait_s) not in (int, float) or end_wait_s not in (15, 30):
            raise ValueError("observed end wait must be 15 or 30 seconds")
        if mode != "request_total" and end_wait_s != 15:
            raise ValueError("TTFT keeps its independent fixed 12 second end wait")
        self.end_wait_s = end_wait_s
        super().__init__(ctx, params)
        self.mode, self.observation_deadline = mode, deadline
        self.observer_error = None

    def issue(self, rid, clock):
        record = super().issue(rid, clock)
        with self._lock:
            record["rpc_observation"] = {}
        return record

    def _join_window(self, entry, seconds):
        deadline = self.observation_deadline
        entry["thread"].join(min(seconds, deadline.remaining()))
        deadline.check()
        return not entry["thread"].is_alive()

    def _start_consumer(self, entry, end):
        if entry["thread"] is not None or entry["record"]["schedule"]["status"] != "OK":
            return
        deadline = self.observation_deadline
        record = entry["record"]
        if self.cancelled:
            self.observer_error = RuntimeError(
                "observed request cancelled before stream startup"
            )
            self.update(
                record,
                rpc_observation=dict(
                    mode=self.mode,
                    complete=False,
                    legacy_success=False,
                    cancelled_before_start=True,
                ),
            )
            return
        observation = dict(
            mode=self.mode,
            operation_started_s=record["issued_s"],
            complete=False,
            legacy_success=False,
        )
        try:
            # Legacy start_stream creates the RPC on the caller before starting
            # the consumer. The observation begins only after thread.start returns.
            call = self._open_stream(entry, min(end, deadline.expires_at))
            entry["opened_call"] = call
            with self._lock:
                entry["call"] = call
                if self.cancelled:
                    call.cancel()
                    raise RuntimeError(
                        "observed request cancelled during stream startup"
                    )
            super()._start_consumer(entry, min(end, deadline.expires_at))
            observation["observer_started_s"] = self.ctx.clock()
            if self.mode == "stream_ttft":
                first_end = observation["observer_started_s"] + 12.0
                while (
                    record["stream"]["first_output_s"] is None
                    and self.ctx.clock() < first_end
                ):
                    # Match the old polling order, including a final 20ms sleep
                    # that can cross the first-output observation boundary.
                    deadline.sleep(0.02)
                deadline.check()
                observation["first_observed_s"] = self.ctx.clock()
                got_first = record["stream"]["first_output_s"] is not None
                observation["got_first"] = got_first
                if got_first:
                    observation["end_wait_started_s"] = self.ctx.clock()
                    ended = self._join_window(entry, 12.0)
                else:
                    ended = False
            else:
                got_first = None
                observation["end_wait_started_s"] = self.ctx.clock()
                ended = self._join_window(entry, self.end_wait_s)
            observation["ended_in_window"] = ended
            observation["end_observed_s"] = self.ctx.clock()
            if not ended:
                # Grace is cleanup time. It cannot repair TTFT's failed end
                # verdict, though old run_one_request checks final completed
                # after this grace and ignores wait_end's false return.
                self.cancel("legacy_observation_window_expired")
                if not self._join_window(entry, 5.0):
                    raise RuntimeError(
                        "observed consumer did not exit within cleanup grace"
                    )
            operation_ended_s = self.ctx.clock()
            self._await_consumer(entry, deadline)
            status = record["stream"]["status"]
            # The old StreamHandle suppresses client CANCELLED as snap.error;
            # retain the transport status itself in the ordinary record.
            no_legacy_error = status in ("OK", "CANCELLED")
            success = (
                (got_first and ended and no_legacy_error)
                if self.mode == "stream_ttft"
                else (record.get("business_finished") is True and no_legacy_error)
            )
            observation.update(
                legacy_success=bool(success),
                complete=True,
                operation_ended_s=operation_ended_s,
            )
        except Exception as exc:
            observation["observer_error"] = repr(exc)
            self.observer_error = exc
        finally:
            self.update(record, rpc_observation=observation)

    def submit(self, deadline):
        if self.cancelled:
            raise RuntimeError("observed request cancelled before Schedule")
        super().submit(deadline)
        if self.observer_error is not None:
            raise self.observer_error
        # A rejected Schedule has no consumer; it is a failed legacy sample,
        # not a fabricated first-output or completion observation.
        for entry in self.entries:
            record = entry["record"]
            if not record["rpc_observation"]:
                self.update(
                    record,
                    rpc_observation=dict(
                        mode=self.mode,
                        operation_started_s=record["issued_s"],
                        operation_ended_s=self.ctx.clock(),
                        complete=True,
                        legacy_success=False,
                        schedule_rejected=True,
                    ),
                )
        self.persist()
