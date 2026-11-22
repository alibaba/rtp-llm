package org.flexlb.mockengine;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.LockSupport;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.grpc.EngineStatusConverter;

/** Single-Master replay of independent Worker request/reply streams, using wall time. */
final class ReplayFeedback implements AutoCloseable {
    private static final long MS = 1_000_000;
    private final List<Lane> lanes = new ArrayList<>();
    private final CountDownLatch startGate = new CountDownLatch(1);
    private final AtomicReference<Throwable> failure = new AtomicReference<>();
    private final PrefillRecoveryReplay replay;
    private final boolean degraded;
    private final long transportNs;
    private volatile long origin;
    private volatile boolean stopping;

    ReplayFeedback(PrefillRecoveryReplay replay, boolean degraded, long transportMs) {
        this.replay = replay;
        this.degraded = degraded;
        transportNs = transportMs * MS;
        for (RoleType role : List.of(RoleType.PREFILL, RoleType.DECODE)) {
            var engines = role == RoleType.PREFILL ? replay.h.prefillEngines : replay.h.decodeEngines;
            for (int i = 0; i < engines.size(); i++) {
                for (boolean cache : List.of(false, true)) {
                    lanes.add(new Lane(role, i, engines.size(), cache, engines.get(i)));
                }
            }
        }
    }

    long start() {
        for (Lane lane : lanes) {
            lane.thread = Thread.ofPlatform().daemon().name(lane.name()).start(lane);
        }
        origin = System.nanoTime();
        startGate.countDown();
        return origin;
    }

    void check() {
        if (failure.get() != null) throw new IllegalStateException("Worker feedback failed", failure.get());
    }

    private void until(long due) {
        long remaining;
        while (!stopping && (remaining = due - System.nanoTime()) > 0) LockSupport.parkNanos(remaining);
    }

    private final class Lane implements Runnable {
        final RoleType role;
        final int worker, count;
        final boolean cache;
        final WorkerEndpoint endpoint;
        final JavaMockEngineCluster.FastRpcService engine;
        Thread thread;
        // Each record is start, captured, apply-start, success, response-version, membership-change.
        long[] records = new long[6 * 512];
        int used;

        Lane(RoleType role, int worker, int count, boolean cache, JavaMockEngineCluster.FastRpcService engine) {
            this.role = role;
            this.worker = worker;
            this.count = count;
            this.cache = cache;
            this.engine = engine;
            endpoint = role == RoleType.PREFILL ? replay.h.prefillEndpoint(worker) : replay.h.decodeEndpoint(worker);
        }
        String name() { return role.name() + "-" + (cache ? "cache" : "status") + "-" + worker; }

        long period(long now) {
            if (!cache) return 20_500_000L;
            if (role == RoleType.DECODE) return 1000 * MS;
            if (degraded) return 40_450_000L;
            long second = (now - origin) / (1000 * MS);
            int transition = replay.realClock ? 400 : replay.baseline;
            if (second < transition) return 49 * MS;
            return second < transition + 60 ? 41_800_000L : 40_450_000L;
        }

        @Override public void run() {
            try {
                startGate.await();
                long due = origin + period(origin) * worker / count;
                long lastVersion = -1;
                while (!stopping) {
                    until(due);
                    if (stopping) break;
                    long start = System.nanoTime();
                    var status = endpoint.getStatus();
                    ReplayPrefillWorker.Delta delta = null;
                    Runnable apply;
                    long version;
                    if (cache) {
                        // Capture capacity/version and the lossless local membership delta before transport.
                        EngineRpcService.CacheStatusPB response;
                        if (role == RoleType.PREFILL) {
                            var physical = replay.observed.get(worker);
                            synchronized (physical.cache) {
                                response = cacheMetadata();
                                delta = physical.captureDelta();
                            }
                        } else {
                            response = cacheMetadata();
                        }
                        version = response.getVersion();
                        var metadata = EngineStatusConverter.convertToCacheStatus(response);
                        var captured = delta;
                        apply = () -> {
                            status.lock.lock();
                            try {
                                if (!replay.h.endpointRegistry.isCurrentStatus(role, status.getIpPort(), status)
                                        || !status.isActiveGeneration()) throw new IllegalStateException(name());
                                status.publishCacheStatus(metadata);
                                if (captured != null) {
                                    captured.apply(replay.index, endpoint.ipPort());
                                    status.publishCacheIndexedVersion(metadata.getVersion());
                                }
                                status.recordSuccessfulCachePoll();
                            } finally {
                                status.lock.unlock();
                            }
                            if (captured != null && captured.touched()) {
                                replay.h.endpointRegistry.publishCacheAvailability(status);
                            }
                        };
                    } else {
                        apply = replay.h.captureReplayStatus(engine, role);
                        version = -1;
                    }
                    long captured = System.nanoTime();
                    until(captured + transportNs);
                    if (stopping) break;
                    long applyStart = System.nanoTime();
                    apply.run();
                    long success = System.nanoTime();
                    if (!cache) version = status.appliedStatusCursor().statusVersion();
                    if (version < lastVersion || (!cache && version == lastVersion)) {
                        throw new IllegalStateException("out-of-order " + name() + " response " + version);
                    }
                    lastVersion = version;
                    if (used == records.length) records = Arrays.copyOf(records, used * 2);
                    records[used++] = start - origin;
                    records[used++] = captured - origin;
                    records[used++] = applyStart - origin;
                    records[used++] = success - origin;
                    records[used++] = version;
                    records[used++] = delta != null && delta.changed() ? 1 : 0;
                    long interval = period(success);
                    due += interval;
                    if (due < success) due += ((success - due + interval - 1) / interval) * interval;
                }
            } catch (Throwable ex) {
                failure.compareAndSet(null, ex);
            }
        }

        EngineRpcService.CacheStatusPB cacheMetadata() {
            return MockEngineTestSupport.unary(observer -> engine.getCacheStatus(
                    EngineRpcService.CacheVersionPB.getDefaultInstance(), observer));
        }
    }

    @Override public void close() throws Exception {
        stopping = true;
        for (Lane lane : lanes) if (lane.thread != null) LockSupport.unpark(lane.thread);
        for (Lane lane : lanes) if (lane.thread != null) lane.thread.join();
        check();
        String file = System.getProperty("feedbackFile");
        if (file == null) return;
        try (var out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(Path.of(file))))) {
            out.writeInt(lanes.size());
            for (Lane lane : lanes) {
                out.writeUTF(lane.name());
                out.writeInt(lane.used / 6);
                for (int i = 0; i < lane.used; i++) out.writeLong(lane.records[i]);
            }
        }
    }
}
