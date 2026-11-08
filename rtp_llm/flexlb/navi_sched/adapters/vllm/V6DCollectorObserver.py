"""Adapter observation of native V6D object events; no storage operations."""

import functools
import logging
import threading

logger = logging.getLogger(__name__)


class V6DCollectorObserver:
    """Attach a KV-only C++ Collector to one node's existing V6D peer.

    The native methods keep their return values, exceptions and storage work.
    Only affected keys are checked after events. The complete resident view is
    read once at attachment, while holding the peer's existing lock.
    """

    def __init__(self, peer, collector, *, pp_shard_count, invalid_object_id):
        if type(pp_shard_count) is not int or pp_shard_count <= 1:
            raise ValueError("V6D Collector observation requires pipeline parallelism")
        self._peer = peer
        self._collector = collector
        self._invalid_object_id = invalid_object_id
        self._restore = []
        self._status_restore = []
        self._status_lock = threading.Lock()
        self._last_peer_read_available = None
        self._stopped = False
        self._failed = False

    def install(self):
        peer = self._peer
        if getattr(peer, "_navi_sched_collector_observer", None) is not None:
            raise RuntimeError("V6D peer already has a Collector observer")
        sync_keys = {
            "_add_object_info": self._first_key,
            "_track_vineyard_add": self._first_key,
            "_add_creating_objects": self._key_list,
            "_add_fetching_objects": self._key_list,
            "_remove_creating_objects": self._key_list,
            "_remove_fetching_objects": self._key_list,
            "_del_object_info": self._key_list,
            "mark_evicted": self._first_key,
            "_track_objects_in_local_vineyard": self._object_keys,
        }
        required = (
            *sync_keys,
            "_track_vineyard_remove",
            "seal",
            "discard",
            "acquire",
            "update_meta",
        )
        for name in required:
            if not callable(getattr(peer, name, None)):
                raise TypeError(f"V6D peer lacks required observation hook: {name}")
        try:
            with peer._lock:
                objects = list(
                    self._resident_identities(tuple(peer._object_info)).items()
                )
                if not self._collector.replace_resident_objects(objects):
                    raise RuntimeError("Collector rejected initial V6D cache view")
                for name, keys in sync_keys.items():
                    self._wrap_sync(name, keys)
                self._wrap_sync("_track_vineyard_remove", self._first_key, removed=True)
                for name in ("seal", "discard"):
                    self._wrap_async(name, self._lease_keys)
                self._wrap_async("acquire", self._key_list, returns_objects=True)
                self._wrap_async("update_meta", self._key_list)
                for name in ("stop", "astop"):
                    if callable(getattr(peer, name, None)):
                        self._wrap_shutdown(name, asynchronous=name == "astop")
                self._install_status_observers()
                peer._navi_sched_collector_observer = self
        except Exception:
            self.stop()
            raise
        return self

    def _native_peer_read_available(self):
        peer = self._peer
        if not peer.config.tair_kvcm_p2p_enabled:
            return False
        store = peer._tair_kvcm_store
        if store is None:
            return False
        mode = peer._tracker.mode
        mode_type = type(mode)
        return bool(
            mode in (mode_type.DUAL_WRITE_READ_KVCM, mode_type.KVCM_ONLY)
            and peer._is_tair_kvcm_report_event_ready()
            and peer._is_tair_kvcm_cache_transfer_ready()
            and store.circuit_breaker_state == "closed"
        )

    def _observe_status(self):
        if self._stopped or self._failed:
            return
        try:
            with self._status_lock:
                if self._stopped or self._failed:
                    return
                available = self._native_peer_read_available()
                if available == self._last_peer_read_available:
                    return
                if not self._collector.report_peer_cache_read_available(available):
                    raise RuntimeError("Collector rejected native peer read state")
                self._last_peer_read_available = available
        except Exception:
            self._disable()

    def _wrap_status(self, owner, name, *, state_attribute=None):
        original = getattr(owner, name)
        if not callable(original):
            raise TypeError(f"V6D peer lacks native status hook: {name}")

        @functools.wraps(original)
        def observed(*args, **kwargs):
            before = getattr(owner, state_attribute) if state_attribute else None
            try:
                return original(*args, **kwargs)
            finally:
                if state_attribute is None or getattr(owner, state_attribute) != before:
                    self._observe_status()

        self._status_restore.append((owner, name, original, observed))
        setattr(owner, name, observed)

    def _install_status_observers(self):
        peer = self._peer
        if peer.config.tair_kvcm_p2p_enabled and peer._tair_kvcm_store is not None:
            client = peer._tair_kvcm_store._client
            self._wrap_status(peer._tracker, "update_mode")
            self._wrap_status(peer._tair_kvcm_store.ready_event, "set")
            for name in ("init_report_event_runtime", "init_transfer_runtime"):
                self._wrap_status(client, name)
            breaker = client._circuit_breaker
            for name in ("record_success", "record_failure"):
                # Native successes are frequent. Only a state transition needs
                # another readiness read and an eventual stream update.
                self._wrap_status(breaker, name, state_attribute="_state")
        self._observe_status()
        if self._failed:
            raise RuntimeError("failed to observe native peer read state")

    @staticmethod
    def _argument(args, kwargs, name):
        return args[0] if args else kwargs.get(name)

    def _first_key(self, args, kwargs):
        value = self._argument(args, kwargs, "object_key")
        return () if value is None else (value,)

    def _key_list(self, args, kwargs):
        return tuple(self._argument(args, kwargs, "object_keys") or ())

    def _object_keys(self, args, kwargs):
        return tuple(
            obj.key
            for obj in self._argument(args, kwargs, "objects") or ()
            if obj is not None
        )

    def _lease_keys(self, args, kwargs):
        lease_id = self._argument(args, kwargs, "lease_id")
        lease = self._peer._active_leases.get(lease_id)
        return tuple(lease.lease.object_key) if lease is not None else ()

    def _capture(self, keys, args, kwargs):
        if self._stopped or self._failed:
            return ()
        try:
            with self._peer._lock:
                return keys(args, kwargs)
        except Exception:
            self._disable()
            return ()

    def _disable(self):
        if self._failed:
            return
        self._failed = True
        logger.exception("V6D Collector observation failed; withdrawing its stream")
        try:
            self._collector.stop()
        except Exception:
            logger.exception("failed to stop V6D Collector after observation failure")

    def _resident_identities(self, keys):
        eligible = []
        for key in keys:
            info = self._peer._object_info.get(key)
            if (
                info is not None
                and info.tier == "vineyard"
                and key not in self._peer._object_to_creating_lease
                and key not in self._peer._object_fetching_keys
            ):
                eligible.append(key)
        ids = (
            self._peer._vineyard_peer.vineyard_server.get_names(list(eligible), "")
            if eligible
            else ()
        )
        if len(ids) != len(eligible):
            raise RuntimeError("incomplete native Vineyard name lookup")
        return {
            key: self._peer._object_info[key].meta.get("navi_sched_cache", "")
            for key, object_id in zip(eligible, ids, strict=True)
            if object_id.id != self._invalid_object_id
        }

    def _observe(self, keys, *, removed=False):
        if self._stopped or self._failed:
            return
        try:
            with self._peer._lock:
                keys = tuple(dict.fromkeys(key for key in keys if key is not None))
                if removed:
                    for key in keys:
                        if not self._collector.object_removed(key):
                            raise RuntimeError("Collector rejected V6D object removal")
                    return
                resident = self._resident_identities(keys)
                for key in keys:
                    if key not in resident:
                        accepted = self._collector.object_removed(key)
                    else:
                        accepted = self._collector.object_resident(key, resident[key])
                    if not accepted:
                        raise RuntimeError("Collector rejected V6D object observation")
        except Exception:
            # Observation must never replace a native storage error or result.
            # Closing the stream makes rank0 retract this shard's cache view.
            self._disable()

    def _wrap_sync(self, name, keys, *, removed=False):
        peer = self._peer
        original = getattr(peer, name)

        @functools.wraps(original)
        def observed(*args, **kwargs):
            affected = self._capture(keys, args, kwargs)
            try:
                return original(*args, **kwargs)
            finally:
                self._observe(affected, removed=removed)

        self._restore.append((name, original, observed))
        setattr(peer, name, observed)

    def _wrap_async(self, name, keys, *, returns_objects=False):
        peer = self._peer
        original = getattr(peer, name)

        @functools.wraps(original)
        async def observed(*args, **kwargs):
            affected = self._capture(keys, args, kwargs)
            result = None
            try:
                result = await original(*args, **kwargs)
                return result
            finally:
                if returns_objects and result is not None:
                    try:
                        affected += tuple(
                            obj.key for obj in result[1] if obj is not None
                        )
                    except Exception:
                        self._disable()
                self._observe(affected)

        self._restore.append((name, original, observed))
        setattr(peer, name, observed)

    def _close_safely(self):
        try:
            self.stop()
        except Exception:
            logger.exception("failed to close V6D Collector during native shutdown")

    def _wrap_shutdown(self, name, *, asynchronous):
        original = getattr(self._peer, name)
        if asynchronous:

            @functools.wraps(original)
            async def observed(*args, **kwargs):
                try:
                    return await original(*args, **kwargs)
                finally:
                    self._close_safely()

        else:

            @functools.wraps(original)
            def observed(*args, **kwargs):
                try:
                    return original(*args, **kwargs)
                finally:
                    self._close_safely()

        self._restore.append((name, original, observed))
        setattr(self._peer, name, observed)

    def stop(self):
        self._stopped = True
        with self._peer._lock:
            for name, original, observed in reversed(self._restore):
                if getattr(self._peer, name) is observed:
                    setattr(self._peer, name, original)
            self._restore.clear()
            for owner, name, original, observed in reversed(self._status_restore):
                if getattr(owner, name) is observed:
                    setattr(owner, name, original)
            self._status_restore.clear()
            if getattr(self._peer, "_navi_sched_collector_observer", None) is self:
                del self._peer._navi_sched_collector_observer
        self._collector.stop()
