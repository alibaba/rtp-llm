"""Restart the backend subtree in place when it dies, keeping frontends and the resident store.

start_server's ProcessManager drives this from its monitor loop on the main thread: the
backend arms PR_SET_PDEATHSIG against the thread that spawned it, so respawning from a
worker thread would kill the new backend as soon as that thread exits.

A row that spans several machines restarts together. Each machine's start_server outlives
its backend generations, so part0's start_server hosts a small TCPStore through which the
row broadcasts a failure, waits until every machine has emptied its old process group, and
lets part0 alone decide the budget and release the next generation. Rank bootstrap is
untouched: once the barrier holds, no rank of the old generation is left to serve or join
the bootstrap store that the new rank 0 creates.
"""

import enum
import logging
import os
import signal
import threading
import time
from datetime import timedelta
from multiprocessing.process import BaseProcess
from typing import Callable, Dict, List, Optional, Tuple

# part0 port = master_server_port - 3; the bootstrap, NCCL and DP layouts never use it.
ROW_COORDINATOR_PORT_OFFSET = 3
DEFAULT_CLEANUP_TIMEOUT_S = 60.0
DEFAULT_HEARTBEAT_INTERVAL_S = 1.0
DEFAULT_HEARTBEAT_TIMEOUT_S = 30.0
DEFAULT_BARRIER_TIMEOUT_S = 180.0
DEFAULT_HEALTHY_RESET_S = 600.0
BACKOFF_BASE_S = 10.0
BACKOFF_MAX_S = 300.0
POLL_INTERVAL_S = 0.2
# A torch TCPStore call can block indefinitely on a stalled peer (set_timeout does not
# bound add/check), so every control-path call runs with this hard bound.
DEFAULT_STORE_OP_TIMEOUT_S = 10.0
# A TCPStore connect cannot be interrupted either (a pending SIGTERM is handled only after it
# returns), so the first connection is made in attempts whose torch timeout is at most this;
# each attempt can still overrun it by torch's own retry delay.
CONNECT_ATTEMPT_S = 2.0


class RestartOutcome(enum.Enum):
    RESTARTED = "restarted"  # a new backend generation is ready
    SHUTDOWN = "shutdown"  # this start_server must stop; take the normal stop path
    GIVE_UP = "give_up"  # behave as without restart: failure shutdown, exit 1


class GiveUp(Exception):
    pass


class ShutdownRequested(Exception):
    pass


class RowStoreError(Exception):
    """The row's coordination store failed or stalled; the row can no longer restart."""


def process_group_members(pgid: int, proc_root: str = "/proc") -> List[int]:
    """Live members of a process group. Zombies hold no resources and are ignored."""
    members = []
    for entry in os.listdir(proc_root):
        if not entry.isdigit():
            continue
        try:
            with open(os.path.join(proc_root, entry, "stat")) as f:
                stat = f.read()
        except OSError:
            continue
        # comm may contain spaces or parentheses; fields resume after the last ')'.
        fields = stat[stat.rfind(")") + 2 :].split()
        if len(fields) < 3:
            continue
        state, pgrp = fields[0], int(fields[2])
        if pgrp == pgid and state != "Z":
            members.append(int(entry))
    return members


def kill_process_group(
    pgid: int,
    timeout: float,
    members: Callable[[int], List[int]] = process_group_members,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
    """SIGKILL a process group and wait until it has no live member."""
    if pgid <= 1 or pgid == os.getpgrp():
        logging.error("refusing to kill process group %s (own group or invalid)", pgid)
        return False
    deadline = clock() + timeout
    while True:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            logging.error("no permission to kill process group %s", pgid)
            return False
        alive = members(pgid)
        if not alive:
            return True
        if clock() >= deadline:
            logging.error(
                "process group %s still has live members %s after %.0fs",
                pgid,
                alive,
                timeout,
            )
            return False
        sleep(POLL_INTERVAL_S)


def _tcp_store(host: str, port: int, is_master: bool, timeout_s: float):
    from torch.distributed import TCPStore

    return TCPStore(
        host_name=host,
        port=port,
        world_size=None,
        is_master=is_master,
        wait_for_workers=False,
        timeout=timedelta(seconds=timeout_s),
    )


class _BoundedStore:
    """Runs store calls on a daemon thread and gives up on any call slower than the bound.

    After one failure the store is treated as gone: later calls fail fast instead of
    queueing behind a call that may never return.
    """

    def __init__(self, store, timeout_s: float):
        self._store = store
        self._timeout_s = timeout_s
        self._broken = False

    def __call__(self, op: str, *args):
        if self._broken:
            raise RowStoreError(f"row store unusable; {op} not attempted")
        result: Dict[str, object] = {}
        done = threading.Event()

        def run():
            try:
                result["value"] = getattr(self._store, op)(*args)
            except BaseException as e:  # reported to the caller below
                result["error"] = e
            done.set()

        threading.Thread(target=run, name=f"row_store_{op}", daemon=True).start()
        if not done.wait(self._timeout_s):
            self._broken = True
            raise RowStoreError(
                f"row store {op} did not return in {self._timeout_s:.0f}s"
            )
        if "error" in result:
            self._broken = True
            cause = (str(result["error"]).splitlines() or [""])[0]
            raise RowStoreError(f"row store {op} failed: {cause}")
        return result["value"]


class RowCoordinator:
    """Coordinates backend generations across the machines of one row.

    With a single machine there is nobody to coordinate with and every method reduces to
    local bookkeeping. With several machines, part0 hosts the store. Keys:

      stop                a machine is stopping normally: the whole row stops normally.
                          It takes precedence over every failure key below.
      gen                 counter, advanced only by part0 when the next generation may start
      fail/<g>            a machine saw generation g die
      clean/<g>/<node>    that machine's generation-g process group is empty
      ready/<g>/<node>    that machine's generation-g backend is ready
      giveup              part0 decided the row stops restarting; everyone fails as before
      hb/<node>           heartbeat counter

    A dedicated thread with its own connection beats and reads every key the monitor
    loop needs, so a monitor tick never touches the network. The thread blocking in a
    stalled call shows up as a stale beat, which counts as losing the row. Writes and
    restart-round waits on the main thread go through a hard per-call bound.
    """

    def __init__(
        self,
        num_nodes: int,
        node_index: int,
        host: str = "",
        port: int = 0,
        connect_timeout_s: float = 600.0,
        heartbeat_interval_s: float = DEFAULT_HEARTBEAT_INTERVAL_S,
        heartbeat_timeout_s: float = DEFAULT_HEARTBEAT_TIMEOUT_S,
        barrier_timeout_s: float = DEFAULT_BARRIER_TIMEOUT_S,
        store_op_timeout_s: float = DEFAULT_STORE_OP_TIMEOUT_S,
        store_factory: Callable[[str, int, bool, float], object] = _tcp_store,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.num_nodes = num_nodes
        self.node_index = node_index
        self.is_leader = node_index == 0
        self.generation = 0
        # Leader/single only: when every machine reported the current generation ready.
        self.row_ready_at: Optional[float] = None
        self._host = host
        self._port = port
        self._connect_timeout_s = connect_timeout_s
        self._hb_interval = heartbeat_interval_s
        self._hb_timeout = heartbeat_timeout_s
        self._barrier_timeout = barrier_timeout_s
        self._op_timeout = store_op_timeout_s
        self._store_factory = store_factory
        self._clock = clock
        self._store: Optional[_BoundedStore] = None
        self._hb_thread: Optional[threading.Thread] = None
        self._hb_stop = threading.Event()
        # Observed by the heartbeat thread (or set locally).
        self._last_beat: Optional[float] = None
        self._peer_lost = False
        # Why the row was given up for lost, logged once.
        self.lost_reason: Optional[str] = None
        self._stop_seen = False
        self._stop_marked = False
        self._giveup_seen = False
        self._fail_seen_gen = -1
        self._ready_gen = -1

    @property
    def multi_node(self) -> bool:
        return self.num_nodes > 1

    def start(self, should_abort: Callable[[], bool] = lambda: False) -> None:
        """Connect to the row store and start beating.

        Raises ShutdownRequested when should_abort() turns true while waiting for part0,
        and RowStoreError when the store cannot be reached in time.
        """
        if not self.multi_node:
            return
        logging.info(
            "row restart coordinator: node %d/%d, store %s:%d (%s)",
            self.node_index,
            self.num_nodes,
            self._host,
            self._port,
            "host" if self.is_leader else "client",
        )
        raw = self._connect(should_abort)
        self._store = _BoundedStore(raw, self._op_timeout)
        self._store("add", f"hb/{self.node_index}", 1)
        self._last_beat = self._clock()
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop, name="row_restart_heartbeat", daemon=True
        )
        self._hb_thread.start()

    def _connect(self, should_abort: Callable[[], bool]):
        """Reach the row store, letting a stop request end the wait between short attempts.

        A stop that arrives during the attempt that connects is seen right after it, before
        any store or heartbeat exists. part0 hosts the store on this machine, so its single
        attempt either works at once or shows a real fault (port taken, wrong address). The
        other machines retry until part0 is up, all within one deadline: the attempts do not
        extend the startup bound. A failed attempt leaves no store or thread behind.
        """
        deadline = self._clock() + self._connect_timeout_s
        while True:
            if should_abort():
                raise ShutdownRequested()
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise RowStoreError(
                    f"row store {self._host}:{self._port} not reachable within "
                    f"{self._connect_timeout_s:.0f}s"
                )
            try:
                raw = self._store_factory(
                    self._host,
                    self._port,
                    self.is_leader,
                    min(CONNECT_ATTEMPT_S, remaining),
                )
            except Exception as e:
                if self.is_leader:
                    cause = (str(e).splitlines() or [type(e).__name__])[0]
                    raise RowStoreError(
                        f"row store could not be hosted: {cause}"
                    ) from e
                time.sleep(
                    POLL_INTERVAL_S
                )  # an attempt can also fail at once (e.g. no route)
                continue
            if should_abort():  # the stop arrived during the attempt that connected
                del raw
                raise ShutdownRequested()
            # The attempt length must not become the store's default operation timeout.
            raw.set_timeout(timedelta(seconds=self._connect_timeout_s))
            return raw

    def close(self) -> None:
        self._hb_stop.set()
        if self._hb_thread is not None:
            self._hb_thread.join(timeout=self._hb_interval * 2 + 1)

    def _mark_lost(self, reason: str) -> None:
        """Give the row up for lost; the first reason is the one logged and reported."""
        if self.lost_reason is None:
            self.lost_reason = reason
            last = (
                "never"
                if self._last_beat is None
                else f"{self._clock() - self._last_beat:.0f}s ago"
            )
            logging.error(
                "row restart: the row is lost at generation %d: %s (last completed beat: %s)",
                self.generation,
                reason,
                last,
            )
        self._peer_lost = True

    def _heartbeat_loop(self) -> None:
        try:
            store = self._store_factory(
                self._host, self._port, False, self._connect_timeout_s
            )
        except Exception as e:
            cause = (str(e).splitlines() or [type(e).__name__])[0]
            self._mark_lost(f"the row coordinator is unreachable: {cause}")
            return
        peers: Dict[int, Tuple[int, float]] = {}
        while not self._hb_stop.is_set():
            g = self.generation
            try:
                self._beat_once(store, g, peers)
            except Exception as e:
                cause = (str(e).splitlines() or [type(e).__name__])[0]
                self._mark_lost(f"the row coordinator is unreachable: {cause}")
                return
            self._hb_stop.wait(self._hb_interval)

    def _beat_once(self, store, g: int, peers: Dict[int, Tuple[int, float]]) -> None:
        store.add(f"hb/{self.node_index}", 1)
        if store.check(["stop"]):
            self._stop_seen = True
        if store.check(["giveup"]):
            self._giveup_seen = True
        if store.check([f"fail/{g}"]):
            self._fail_seen_gen = g
        if self._ready_gen != g and store.check(
            [f"ready/{g}/{n}" for n in range(self.num_nodes)]
        ):
            self._ready_gen = g
            if self.is_leader and self.generation == g:
                self.row_ready_at = self._clock()
        now = self._clock()
        for peer in range(self.num_nodes):
            if peer == self.node_index:
                continue
            value = store.add(f"hb/{peer}", 0)
            previous = peers.get(peer)
            if previous is None or value != previous[0]:
                peers[peer] = (value, now)
            elif value > 0 and now - previous[1] > self._hb_timeout:
                self._mark_lost(
                    f"node {peer} stopped beating for {now - previous[1]:.0f}s"
                )
        self._last_beat = now

    def stop_requested(self) -> bool:
        return self._stop_seen

    def peer_lost(self) -> bool:
        """A machine of the row, or part0's store, is gone (and the row is not stopping)."""
        if self._stop_seen:
            return False
        if (
            self._last_beat is not None
            and self._clock() - self._last_beat > self._hb_timeout
        ):
            self._mark_lost(
                f"this machine's calls to the row coordinator have not completed for "
                f"{self._clock() - self._last_beat:.0f}s"
            )
        return self._peer_lost

    def mark_stop(self) -> None:
        """This machine stops normally; every machine of the row stops normally too."""
        if self._stop_marked:
            return
        self._stop_marked = True
        self._stop_seen = True
        if self._store is not None:
            try:
                self._store("set", "stop", str(self.node_index))
            except RowStoreError as e:
                logging.warning("row restart: failed to publish stop: %s", e)

    def row_failure_pending(self) -> bool:
        """The current generation died elsewhere in the row, or a machine is gone."""
        if not self.multi_node or self._stop_seen:
            return False
        return (
            self.peer_lost()
            or self._giveup_seen
            or self._fail_seen_gen == self.generation
        )

    def row_ready(self) -> bool:
        return self._ready_gen == self.generation

    def report_failure(self, reason: str) -> None:
        if self._store is not None:
            self._store("set", f"fail/{self.generation}", f"{self.node_index}:{reason}")

    def report_clean(self) -> None:
        if self._store is not None:
            self._store("set", f"clean/{self.generation}/{self.node_index}", "1")

    def report_ready(self) -> None:
        if self._store is None:
            self._ready_gen = self.generation
            self.row_ready_at = self._clock()
        else:
            self._store("set", f"ready/{self.generation}/{self.node_index}", "1")

    def await_next_generation(
        self,
        decide_backoff: Callable[[], Optional[float]],
        should_abort: Callable[[], bool],
        sleep: Callable[[float], None] = time.sleep,
    ) -> int:
        """Block until the next generation may start.

        Raises ShutdownRequested, GiveUp or RowStoreError.
        """
        if self._store is None:
            backoff = decide_backoff()
            if backoff is None:
                raise GiveUp("restart budget exhausted")
            self._wait(backoff, should_abort, sleep)
        elif self.is_leader:
            self._lead_next_generation(decide_backoff, should_abort, sleep)
        else:
            self._follow_next_generation(should_abort, sleep)
        self.generation += 1
        self.row_ready_at = None
        return self.generation

    def _check_row(self, should_abort: Callable[[], bool]) -> None:
        # Normal stop first: it outranks any failure or give-up recorded meanwhile.
        if should_abort() or self._stop_seen:
            raise ShutdownRequested()
        if self._store is None:
            return
        if self._giveup_seen:
            raise GiveUp("part0 gave up restarting the row")
        if self.peer_lost():
            raise GiveUp(f"a machine of the row is gone: {self.lost_reason}")

    def _lead_next_generation(self, decide_backoff, should_abort, sleep) -> None:
        g = self.generation
        clean_keys = [f"clean/{g}/{n}" for n in range(self.num_nodes)]
        deadline = self._clock() + self._barrier_timeout
        try:
            while not self._store("check", clean_keys):
                self._check_row(should_abort)
                if self._clock() >= deadline:
                    raise GiveUp(f"generation {g} was not cleaned on every machine")
                sleep(POLL_INTERVAL_S)
            backoff = decide_backoff()
            if backoff is None:
                raise GiveUp("restart budget exhausted")
            self._wait(backoff, should_abort, sleep)
        except GiveUp:
            try:
                self._store("set", "giveup", str(g))
            except RowStoreError:
                pass
            raise
        self._store("add", "gen", 1)
        logging.info("row restart: released generation %d", g + 1)

    def _follow_next_generation(self, should_abort, sleep) -> None:
        g = self.generation
        while self._store("add", "gen", 0) <= g:
            self._check_row(should_abort)
            sleep(POLL_INTERVAL_S)

    def _wait(self, seconds: float, should_abort, sleep) -> None:
        deadline = self._clock() + seconds
        while True:
            self._check_row(should_abort)
            remaining = deadline - self._clock()
            if remaining <= 0:
                return
            sleep(min(POLL_INTERVAL_S, remaining))


SpawnBackend = Callable[[], Tuple[BaseProcess, Callable[[], bool]]]


class RowEvent(enum.Enum):
    NONE = "none"
    STOP = "stop"  # another machine of the row is stopping normally
    FAILURE = "failure"  # the current generation died elsewhere, or a machine is gone


class _Round(Exception):
    """The generation being started failed; run another restart round."""

    def __init__(self, reason: str, local: bool):
        super().__init__(reason)
        self.reason = reason
        self.local = local


class BackendRestarter:
    """Replaces a dead backend subtree with a new generation built from the same config.

    `spawn_backend` starts a backend process and returns it with its readiness check (the
    startup pipe). `after_ready` runs the same post-readiness steps as the first start
    (startup warmup). The health gate file makes /health answer 503 from the moment the
    old generation dies until every rank of the new one, on every machine of the row, is
    ready.
    """

    def __init__(
        self,
        restart_limit: int,
        spawn_backend: SpawnBackend,
        coordinator: RowCoordinator,
        is_shutdown_requested: Callable[[], bool],
        gate_file: Optional[str] = None,
        after_ready: Optional[Callable[[], None]] = None,
        ready_timeout_s: Optional[float] = 3600.0,
        cleanup_timeout_s: float = DEFAULT_CLEANUP_TIMEOUT_S,
        healthy_reset_s: float = DEFAULT_HEALTHY_RESET_S,
        backoff_base_s: float = BACKOFF_BASE_S,
        kill_group: Callable[[int, float], bool] = kill_process_group,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.restart_limit = restart_limit
        self.coordinator = coordinator
        self._spawn_backend = spawn_backend
        self._is_shutdown_requested = is_shutdown_requested
        self._gate_file = gate_file
        self._after_ready = after_ready
        self._ready_timeout_s = ready_timeout_s
        self._cleanup_timeout_s = cleanup_timeout_s
        self._healthy_reset_s = healthy_reset_s
        self._backoff_base_s = backoff_base_s
        self._kill_group = kill_group
        self._sleep = sleep
        self._clock = clock
        self._armed = False
        self._restarts = 0
        # Backend processes started or inherited and not yet confirmed gone. Whatever
        # happens inside a restart, these go back to the ProcessManager.
        self.live: List[BaseProcess] = []

    @property
    def restarts(self) -> int:
        return self._restarts

    def arm(self) -> None:
        """The first start is complete; failures from now on are restarted."""
        self.coordinator.report_ready()
        self._armed = True

    def on_shutdown(self) -> None:
        self.coordinator.mark_stop()

    def poll(self) -> RowEvent:
        """Called on every healthy monitor tick; reads only state the heartbeat keeps."""
        if self.coordinator.stop_requested():
            return RowEvent.STOP
        if self._armed and self.coordinator.row_failure_pending():
            return RowEvent.FAILURE
        return RowEvent.NONE

    def _abort(self) -> bool:
        return self._is_shutdown_requested()

    def _stopping(self) -> bool:
        return self._abort() or self.coordinator.stop_requested()

    def _decide_backoff(self) -> Optional[float]:
        """Leader only: count one restart of the row and return its backoff, or None."""
        ready_at = self.coordinator.row_ready_at
        if ready_at is not None and self._clock() - ready_at >= self._healthy_reset_s:
            self._restarts = 0
        self._restarts += 1
        if self._restarts > self.restart_limit:
            logging.error(
                "backend restart budget exhausted (%d restarts, limit %d)",
                self._restarts - 1,
                self.restart_limit,
            )
            return None
        return min(self._backoff_base_s * 2 ** (self._restarts - 1), BACKOFF_MAX_S)

    def _set_gate(self, ready: bool) -> None:
        if not self._gate_file:
            return
        if ready:
            with open(self._gate_file, "w") as f:
                f.write("ready\n")
        else:
            try:
                os.remove(self._gate_file)
            except FileNotFoundError:
                pass

    def _clean(self) -> bool:
        """Kill each backend and the process group it leads, and confirm both are gone.

        The backend itself goes first: it may not lead its group yet (still starting),
        and once it is dead it cannot create ranks. Ranks it did create share its group.
        """
        for proc in list(self.live):
            if proc.pid is not None:
                if proc.is_alive():
                    proc.kill()
                proc.join(timeout=self._cleanup_timeout_s)
                if proc.is_alive():
                    logging.error("backend pid %s survived SIGKILL", proc.pid)
                    return False
                if not self._kill_group(proc.pid, self._cleanup_timeout_s):
                    return False
            self.live.remove(proc)
        return True

    def _running(self) -> List[BaseProcess]:
        return [p for p in self.live if p.is_alive()]

    def handle_backend_failure(
        self, old_procs: List[BaseProcess], local: bool
    ) -> Tuple[RestartOutcome, List[BaseProcess]]:
        """Run restart rounds until a generation is ready, a stop arrives or we give up.

        Returns the outcome and the backend processes the ProcessManager must own.
        """
        self.live = list(old_procs)
        if not self._armed:
            return RestartOutcome.GIVE_UP, self.live
        try:
            return self._restart(local)
        except RowStoreError as e:
            logging.error("backend restart given up, row store failed: %s", e)
            return RestartOutcome.GIVE_UP, self.live

    def _restart(self, local: bool) -> Tuple[RestartOutcome, List[BaseProcess]]:
        reason = "backend died" if local else "row failure"
        while True:
            # A normal stop anywhere in the row outranks the failure being handled.
            if self._stopping():
                self.coordinator.mark_stop()
                # A generation that already died is not a failure of this stop: hand
                # back only what is still running, for the normal staged stop.
                return RestartOutcome.SHUTDOWN, self._running()
            if self.coordinator.peer_lost():
                return RestartOutcome.GIVE_UP, self.live
            logging.error(
                "backend generation %d failed (%s); restarting the backend subtree",
                self.coordinator.generation,
                reason,
            )
            if local:
                self.coordinator.report_failure(reason)
            self._set_gate(False)
            if not self._clean():
                return RestartOutcome.GIVE_UP, self.live
            self.coordinator.report_clean()
            try:
                self.coordinator.await_next_generation(
                    self._decide_backoff, self._abort, self._sleep
                )
            except ShutdownRequested:
                self.coordinator.mark_stop()
                return RestartOutcome.SHUTDOWN, []
            except GiveUp as e:
                logging.error("backend restart given up: %s", e)
                return RestartOutcome.GIVE_UP, []
            try:
                return self._start_generation()
            except _Round as next_round:
                reason, local = next_round.reason, next_round.local
            except ShutdownRequested:
                self.coordinator.mark_stop()
                return RestartOutcome.SHUTDOWN, self._running()

    def _start_generation(self) -> Tuple[RestartOutcome, List[BaseProcess]]:
        g = self.coordinator.generation
        logging.info("starting backend generation %d", g)
        try:
            proc, check_ready = self._spawn_backend()
        except Exception as e:  # e.g. OSError from Process.start
            raise _Round(f"backend spawn failed: {e}", local=True)
        self.live.append(proc)
        deadline = (
            None
            if self._ready_timeout_s is None
            else self._clock() + self._ready_timeout_s
        )

        def check_generation(waiting_for: str) -> None:
            if self._stopping():
                raise ShutdownRequested()
            if self.coordinator.row_failure_pending():
                raise _Round(f"row failure while {waiting_for}", local=False)
            if not proc.is_alive():
                raise _Round(f"backend died while {waiting_for}", local=True)
            if deadline is not None and self._clock() >= deadline:
                raise _Round(f"timed out while {waiting_for}", local=True)

        while True:
            check_generation("starting")
            try:
                if check_ready():
                    break
            except Exception as e:  # the startup pipe reports a failed or closed start
                raise _Round(f"backend start failed: {e}", local=True)
            self._sleep(1.0)
        if self._after_ready is not None:
            try:
                self._after_ready()
            except Exception as e:
                raise _Round(f"post-start step failed: {e}", local=True)
        check_generation("finishing the start")
        self.coordinator.report_ready()
        # Traffic returns only when every machine of the row runs this generation.
        while not self.coordinator.row_ready():
            check_generation("waiting for the rest of the row")
            self._sleep(POLL_INTERVAL_S)
        # A stop or failure can arrive while report_ready was writing.
        check_generation("opening the health gate")
        self._set_gate(True)
        logging.info("backend generation %d is ready", g)
        return RestartOutcome.RESTARTED, [proc]
