"""Backend-owned shutdown rendezvous; never asks an unsignalled peer to exit."""

import hashlib
import json
import logging
import time
from datetime import timedelta

from rtp_llm.utils.lifecycle_lease import LifecycleLease

_PREFIX = "rtp_llm_backend_shutdown/"


class ShutdownError(RuntimeError):
    """The instance could not establish a safe, complete shutdown boundary."""


def register_shutdown_member(store, rank, control):
    """Publish a live incarnation after engine construction, not at shutdown."""
    status = control.shutdown_status()
    incarnation = status["worker_incarnation"]
    if not incarnation:
        raise ShutdownError("backend shutdown registration has no incarnation")
    store.set(_PREFIX + f"member/{rank}", incarnation)
    return incarnation


def graceful_backend_shutdown(
    control,
    store,
    rank,
    world_size,
    incarnation,
    timeout_s,
    *,
    clock=time.monotonic,
    sleep=time.sleep,
):
    """Keep TCPStore and model RPC alive until every signalled rank is stopped.

    The instance lease is shared with frontend sleep/wake. It is deliberately
    retained after terminal intent: no new lifecycle operation may start while
    backends are closing their RPC listeners. It disappears with the store.
    A stale lease is never stolen. Failures do not call terminate or release
    resources; the caller reports a failed shutdown to the process supervisor.
    """
    if world_size < 1 or not 0 <= rank < world_size or timeout_s <= 0:
        raise ValueError("invalid shutdown rank, world size or timeout")
    deadline = clock() + timeout_s

    def remaining_ms():
        remaining = deadline - clock()
        if remaining <= 0:
            raise ShutdownError("backend shutdown total deadline exceeded")
        return max(1, int(remaining * 1000))

    def read(keys):
        # A stalled store must not consume its bootstrap timeout on every poll.
        store.set_timeout(timedelta(milliseconds=min(3000, remaining_ms())))
        if not store.check(keys):
            return None
        return [store.get(key).decode("utf-8") for key in keys]

    group_prefix = None
    member_keys = [_PREFIX + f"member/{i}" for i in range(world_size)]

    def wait_for(predicate, phase):
        started = clock()
        while True:
            remaining_ms()
            if group_prefix is not None:
                current_members = read(member_keys)
                if current_members != members:
                    raise ShutdownError(f"backend incarnation changed during {phase}")
                for peer in range(world_size):
                    error = read([group_prefix + f"error/{peer}"])
                    if error:
                        raise ShutdownError(f"peer {peer} failed: {error[0]}")
            result = predicate()
            if result is not None:
                logging.info(
                    "[BackendShutdown] rank=%s phase=%s elapsed_ms=%.3f",
                    rank,
                    phase,
                    (clock() - started) * 1000,
                )
                return result
            sleep(min(0.05, remaining_ms() / 1000))

    def barrier(phase, value="ok"):
        store.set(group_prefix + f"{phase}/{rank}", value)
        keys = [group_prefix + f"{phase}/{i}" for i in range(world_size)]
        return wait_for(lambda: read(keys), phase)

    try:
        # This rank alone has received a shutdown request. Closing its admission
        # does not instruct any other process or machine to terminate.
        control.begin_shutdown()
        members = wait_for(lambda: read(member_keys), "membership")
        if members[rank] != incarnation or len(set(members)) != world_size:
            raise ShutdownError("invalid or stale backend shutdown membership")
        group_id = hashlib.sha256(json.dumps(members).encode()).hexdigest()
        group_prefix = _PREFIX + group_id + "/"
        barrier("intent", incarnation)

        if rank == 0:
            lease = LifecycleLease(store, None, required=True)

            def acquire():
                record, error = lease.acquire("shutdown")
                return record if not error else None

            record = wait_for(acquire, "instance_lease")
            store.set(group_prefix + "authorized", record)
        wait_for(lambda: read([group_prefix + "authorized"]), "authorized")

        states = barrier("state", control.shutdown_status()["state"])
        if len(set(states)) != 1 or states[0] not in ("RUNNING", "SLEEPING"):
            raise ShutdownError(
                f"cannot stop partially transitioned instance: {states}"
            )

        control.drain_shutdown(remaining_ms(), False)
        barrier("drained")
        control.drain_shutdown(remaining_ms(), True)
        barrier("sealed_and_drained")

        if states[0] == "RUNNING":
            rounds = barrier("frozen", str(control.freeze_shutdown()))
            target = max(int(value) for value in rounds)
            if target < 0 or target >= 1 << 63:
                raise ShutdownError("invalid shutdown round")
            control.quiesce_shutdown(target, remaining_ms())
        # SLEEPING is already quiesced; never reload weights merely to exit.
        barrier("quiesced")
        # Retiring an async runner can start deferred cache writeback after the
        # earlier drains. Keep all resources/control channels alive until every
        # rank has also retired these final transfers. Admission stays sealed.
        control.drain_shutdown(remaining_ms(), True)
        barrier("post_quiesce_drained")
        control.terminate_shutdown()
        barrier("terminated")

        # rank 0 owns the bootstrap store. Keep it alive until every peer has
        # consumed the all-terminated result, not just published its own result.
        if rank == 0:
            keys = [group_prefix + f"observed/{i}" for i in range(1, world_size)]
            if keys:
                wait_for(lambda: read(keys), "all_observed")
        else:
            store.set(group_prefix + f"observed/{rank}", "ok")
        logging.info("[BackendShutdown] rank=%s GRACEFUL_SUCCESS", rank)
    except Exception as error:
        if group_prefix is not None:
            try:
                store.set_timeout(timedelta(seconds=1))
                store.set(group_prefix + f"error/{rank}", str(error))
            except Exception:
                logging.exception("failed to publish backend shutdown failure")
        raise ShutdownError(f"rank {rank} graceful shutdown failed: {error}") from error
