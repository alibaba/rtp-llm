package org.flexlb.mockengine;

import io.grpc.ManagedChannel;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.netty.NettyChannelBuilder;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.socket.nio.NioSocketChannel;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.nio.file.Path;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Schedule routing for HA traffic. With a discovery file, the selected Master
 * and healthy slave come from periodic /master/info probes; without it, the
 * older GRPC_TARGETS mode retains its sticky transport-failover behavior.
 *
 * <p>In discovery mode the candidate file plays VipServer's membership role.
 * The routing policy follows the production frontend: prefer a healthy node
 * reporting itself as leader, retain the healthy previous choice when no node
 * claims leadership, and retry a healthy non-leader only after a non-deadline
 * Schedule connection failure. A Schedule response, including a business
 * error, is terminal. The mock dual_standalone topology itself has no Master
 * election or follower forwarding.
 *
 * <p>Per-request attempt targets and retry status are recorded in the request
 * ledger. Discovery snapshots are immutable across an individual request.
 */
final class MasterTargetRouter {

    /** Coarse failure taxonomy for the per-request {@code error_kind} field. */
    enum ErrorKind {
        /** Schedule RPC returned a response (any code — business codes are
         *  classified by the caller, not the router). */
        NONE("none"),
        /** Every eligible target failed before a Schedule response. */
        TRANSPORT("transport"),
        /** gRPC DEADLINE_EXCEEDED — no retry, no switch, no fallback. */
        DEADLINE("deadline"),
        /** Master answered at the gRPC layer but the call failed (INTERNAL,
         *  CANCELLED, ...) or a non-gRPC exception escaped — conservative:
         *  same no-retry boundary as business error codes. */
        BUSINESS("business");

        final String label;

        ErrorKind(String label) {
            this.label = label;
        }
    }

    /** Observed transport attempt; timestamps use one monotonic elapsed interval. */
    static final class ScheduleAttempt {
        final String target;
        final long startedEpochMs;
        final long endedEpochMs;
        final String status;
        final Integer responseCode;

        ScheduleAttempt(String target, long startedEpochMs, long startedNanos,
                String status, Integer responseCode) {
            this.target = target;
            this.startedEpochMs = startedEpochMs;
            this.endedEpochMs = startedEpochMs
                    + TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startedNanos);
            this.status = status;
            this.responseCode = responseCode;
        }
    }

    /** Terminal outcome of one failover-aware Schedule call. */
    static final class ScheduleOutcome {
        /** Master response, or null when no target produced one. */
        final FlexlbScheduleProtocol.FlexlbScheduleResponsePB response;
        /** Actually-served (or last-attempted, when none served) target address. */
        final String lastTarget;
        /** True when the same request was retried on another target. */
        final boolean failover;
        final ErrorKind errorKind;
        /** Transport-layer exception of the last UNAVAILABLE attempt (null otherwise). */
        final Exception failure;
        final List<ScheduleAttempt> attempts;

        ScheduleOutcome(FlexlbScheduleProtocol.FlexlbScheduleResponsePB response,
                String lastTarget, boolean failover, ErrorKind errorKind, Exception failure,
                List<ScheduleAttempt> attempts) {
            this.response = response;
            this.lastTarget = lastTarget;
            this.failover = failover;
            this.errorKind = errorKind;
            this.failure = failure;
            this.attempts = List.copyOf(attempts);
        }
    }

    /** One target's channel pool — same N_CHANNELS + round-robin shape as the legacy single-target path. */
    private static final class TargetPool {
        final String target;
        final ManagedChannel[] channels;
        final FlexlbServiceGrpc.FlexlbServiceBlockingStub[] stubs;
        final AtomicInteger rr = new AtomicInteger();

        TargetPool(String target, ManagedChannel[] channels,
                FlexlbServiceGrpc.FlexlbServiceBlockingStub[] stubs) {
            this.target = target;
            this.channels = channels;
            this.stubs = stubs;
        }

        FlexlbServiceGrpc.FlexlbServiceBlockingStub nextStub() {
            int idx = Math.floorMod(rr.getAndIncrement(), stubs.length);
            return stubs[idx];
        }
    }

    private final List<TargetPool> pools;
    private final Map<String, TargetPool> discoveredPools = new ConcurrentHashMap<>();
    private final MasterRouteDiscovery discovery;
    private final EventLoopGroup eventLoopGroup;
    private final int nChannels;
    /** Index into {@link #pools} of the current sticky target. */
    private volatile int sticky;

    /**
     * Production constructor: builds an N_CHANNELS channel pool per target,
     * with the same channel options as the legacy single-target path in
     * JavaLoadClient.
     */
    MasterTargetRouter(List<String> targets, int nChannels, EventLoopGroup eventLoopGroup) {
        this.discovery = null;
        this.eventLoopGroup = eventLoopGroup;
        this.nChannels = nChannels;
        this.pools = new ArrayList<>(targets.size());
        for (String target : targets) {
            pools.add(newTargetPool(target));
        }
        this.sticky = 0;
    }

    MasterTargetRouter(Path discoveryFile, int nChannels, EventLoopGroup eventLoopGroup) {
        this(MasterRouteDiscovery.fromFile(discoveryFile), nChannels, eventLoopGroup);
        discovery.start();
    }

    private MasterTargetRouter(MasterRouteDiscovery discovery, int nChannels,
            EventLoopGroup eventLoopGroup) {
        this.discovery = discovery;
        this.eventLoopGroup = eventLoopGroup;
        this.nChannels = nChannels;
        this.pools = List.of();
    }

    MasterTargetRouter(MasterRouteDiscovery discovery,
            Map<String, FlexlbServiceGrpc.FlexlbServiceBlockingStub[]> stubs) {
        this(discovery, 0, null);
        for (Map.Entry<String, FlexlbServiceGrpc.FlexlbServiceBlockingStub[]> entry : stubs.entrySet()) {
            discoveredPools.put(entry.getKey(), new TargetPool(entry.getKey(), null, entry.getValue()));
        }
    }

    private TargetPool newTargetPool(String target) {
        ManagedChannel[] channels = new ManagedChannel[nChannels];
        FlexlbServiceGrpc.FlexlbServiceBlockingStub[] stubs =
                new FlexlbServiceGrpc.FlexlbServiceBlockingStub[nChannels];
        for (int i = 0; i < nChannels; i++) {
            ManagedChannel channel = NettyChannelBuilder.forTarget(target)
                    .eventLoopGroup(eventLoopGroup)
                    .channelType(NioSocketChannel.class)
                    .maxInboundMessageSize(16 * 1024 * 1024)
                    .flowControlWindow(1024 * 1024)
                    .keepAliveTime(30, TimeUnit.SECONDS)
                    .keepAliveTimeout(10, TimeUnit.SECONDS)
                    .usePlaintext()
                    .build();
            channels[i] = channel;
            stubs[i] = FlexlbServiceGrpc.newBlockingStub(channel);
        }
        return new TargetPool(target, channels, stubs);
    }

    /**
     * Test constructor: injects stub arrays directly (channels may be null —
     * {@link #shutdown()} skips them).
     */
    MasterTargetRouter(List<String> targets, List<FlexlbServiceGrpc.FlexlbServiceBlockingStub[]> stubs) {
        this.discovery = null;
        this.eventLoopGroup = null;
        this.nChannels = 0;
        this.pools = new ArrayList<>(targets.size());
        for (int i = 0; i < targets.size(); i++) {
            pools.add(new TargetPool(targets.get(i), null, stubs.get(i)));
        }
        this.sticky = 0;
    }

    String stickyTarget() {
        if (discovery != null) {
            return discovery.route().master();
        }
        return pools.get(sticky).target;
    }

    /**
     * One failover-aware Schedule call. See the class javadoc for the decision
     * flow; the method never throws — every failure is folded into the
     * returned {@link ScheduleOutcome}.
     */
    ScheduleOutcome schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB request, long timeoutMs) {
        if (discovery != null) {
            return scheduleDiscovered(request, timeoutMs);
        }
        int start = sticky;
        Exception lastFailure = null;
        List<ScheduleAttempt> attempts = new ArrayList<>();
        for (int attempt = 0; attempt < pools.size(); attempt++) {
            int idx = Math.floorMod(start + attempt, pools.size());
            TargetPool pool = pools.get(idx);
            FlexlbServiceGrpc.FlexlbServiceBlockingStub stub = pool.nextStub()
                    .withDeadlineAfter(timeoutMs, TimeUnit.MILLISECONDS);
            long startedEpochMs = System.currentTimeMillis();
            long startedNanos = System.nanoTime();
            try {
                FlexlbScheduleProtocol.FlexlbScheduleResponsePB response = stub.schedule(request);
                // Any response (code 200 or business error) proves the target
                // alive: the sticky pointer follows it. A business error from
                // the sticky target itself leaves the pointer untouched in
                // practice (idx == sticky) — it only moves when transport
                // failover delivered us to another target first.
                sticky = idx;
                attempts.add(new ScheduleAttempt(pool.target, startedEpochMs, startedNanos,
                        "OK", response.getCode()));
                return new ScheduleOutcome(response, pool.target, attempt > 0, ErrorKind.NONE, null, attempts);
            } catch (StatusRuntimeException e) {
                Status.Code code = e.getStatus().getCode();
                attempts.add(new ScheduleAttempt(pool.target, startedEpochMs, startedNanos,
                        code.name(), null));
                if (code == Status.Code.UNAVAILABLE) {
                    // Event ① (transport-layer unreachable): same-request retry
                    // on the next target — never wait for ZK, never probe.
                    lastFailure = e;
                    continue;
                }
                // DEADLINE_EXCEEDED and every other gRPC status: terminal for
                // this request — no retry, no switch, no direct fallback.
                ErrorKind kind = code == Status.Code.DEADLINE_EXCEEDED
                        ? ErrorKind.DEADLINE : ErrorKind.BUSINESS;
                return new ScheduleOutcome(null, pool.target, attempt > 0, kind, e, attempts);
            } catch (RuntimeException e) {
                attempts.add(new ScheduleAttempt(pool.target, startedEpochMs, startedNanos,
                        "EXCEPTION", null));
                // Non-gRPC failure: conservative — only UNAVAILABLE retries.
                return new ScheduleOutcome(null, pool.target, attempt > 0, ErrorKind.BUSINESS, e, attempts);
            }
        }
        // Every target answered UNAVAILABLE: double connection failure. The
        // sticky pointer stays put — with no probing thread the next request
        // simply repeats the failover chain (observable as failover=true rows
        // while the outage lasts, matching the "fallback <= 1 per request"
        // case-test assertions).
        int lastIdx = Math.floorMod(start + pools.size() - 1, pools.size());
        return new ScheduleOutcome(null, pools.get(lastIdx).target, pools.size() > 1,
                ErrorKind.TRANSPORT, lastFailure, attempts);
    }

    private ScheduleOutcome scheduleDiscovered(
            FlexlbScheduleProtocol.FlexlbScheduleRequestPB request, long timeoutMs) {
        MasterRouteDiscovery.Route route = discovery.route();
        if (route.master() == null) {
            return new ScheduleOutcome(null, "", false, ErrorKind.TRANSPORT, null, List.of());
        }
        List<String> targets = new ArrayList<>();
        targets.add(route.master());
        if (route.slave() != null && !route.slave().equals(route.master())) {
            targets.add(route.slave());
        }
        List<ScheduleAttempt> attempts = new ArrayList<>();
        Exception lastFailure = null;
        for (String target : targets) {
            TargetPool pool = discoveredPools.computeIfAbsent(target, this::newTargetPool);
            FlexlbServiceGrpc.FlexlbServiceBlockingStub stub = pool.nextStub()
                    .withDeadlineAfter(timeoutMs, TimeUnit.MILLISECONDS);
            long startedEpochMs = System.currentTimeMillis();
            long startedNanos = System.nanoTime();
            try {
                FlexlbScheduleProtocol.FlexlbScheduleResponsePB response = stub.schedule(request);
                attempts.add(new ScheduleAttempt(target, startedEpochMs, startedNanos,
                        "OK", response.getCode()));
                return new ScheduleOutcome(response, target, attempts.size() > 1,
                        ErrorKind.NONE, null, attempts);
            } catch (StatusRuntimeException e) {
                Status.Code code = e.getStatus().getCode();
                attempts.add(new ScheduleAttempt(target, startedEpochMs, startedNanos,
                        code.name(), null));
                if (code == Status.Code.DEADLINE_EXCEEDED) {
                    return new ScheduleOutcome(null, target, attempts.size() > 1,
                            ErrorKind.DEADLINE, e, attempts);
                }
                lastFailure = e;
            } catch (RuntimeException e) {
                attempts.add(new ScheduleAttempt(target, startedEpochMs, startedNanos,
                        "EXCEPTION", null));
                lastFailure = e;
            }
        }
        return new ScheduleOutcome(null, targets.get(targets.size() - 1), attempts.size() > 1,
                ErrorKind.TRANSPORT, lastFailure, attempts);
    }

    /** Shuts down every target pool's channels (no-op for test-constructed pools). */
    void shutdown() {
        if (discovery != null) {
            discovery.close();
        }
        List<TargetPool> allPools = new ArrayList<>(pools);
        allPools.addAll(discoveredPools.values());
        for (TargetPool pool : allPools) {
            if (pool.channels == null) {
                continue;
            }
            for (ManagedChannel channel : pool.channels) {
                channel.shutdown();
            }
        }
    }

    /**
     * Maps a Schedule-call exception to the per-request {@code error_kind}
     * label. Used by the legacy single-target path too, so both modes stamp
     * the same taxonomy: UNAVAILABLE → transport, DEADLINE_EXCEEDED →
     * deadline, anything else → business (master-answered-at-gRPC-layer or
     * unknown — never retried either way).
     */
    static String classifyThrowable(Throwable t) {
        if (t instanceof StatusRuntimeException) {
            Status.Code code = ((StatusRuntimeException) t).getStatus().getCode();
            if (code == Status.Code.UNAVAILABLE) {
                return ErrorKind.TRANSPORT.label;
            }
            if (code == Status.Code.DEADLINE_EXCEEDED) {
                return ErrorKind.DEADLINE.label;
            }
        }
        return ErrorKind.BUSINESS.label;
    }
}
