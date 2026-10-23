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
 * Schedule routing from ZK leader discovery. A request uses one discovery snapshot,
 * tries the leader first, and retries a healthy follower only on UNAVAILABLE.
 * Business responses and deadlines are terminal. The candidate file is the sole
 * address source; successful follower retries do not change the discovered leader.
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

    private final Map<String, TargetPool> discoveredPools = new ConcurrentHashMap<>();
    private final MasterRouteDiscovery discovery;
    private final EventLoopGroup eventLoopGroup;
    private final int nChannels;
    MasterTargetRouter(Path discoveryFile, int nChannels, EventLoopGroup eventLoopGroup) {
        this(MasterRouteDiscovery.fromFile(discoveryFile), nChannels, eventLoopGroup);
        discovery.start();
    }

    private MasterTargetRouter(MasterRouteDiscovery discovery, int nChannels,
            EventLoopGroup eventLoopGroup) {
        this.discovery = discovery;
        this.eventLoopGroup = eventLoopGroup;
        this.nChannels = nChannels;
    }

    MasterTargetRouter(MasterRouteDiscovery discovery,
            Map<String, FlexlbServiceGrpc.FlexlbServiceBlockingStub[]> stubs) {
        this(discovery, 0, null);
        for (Map.Entry<String, FlexlbServiceGrpc.FlexlbServiceBlockingStub[]> entry : stubs.entrySet()) {
            discoveredPools.put(entry.getKey(), new TargetPool(entry.getKey(), null, entry.getValue()));
        }
    }

    private TargetPool newTargetPool(String target) {
        if (nChannels < 1) throw new IllegalStateException("no injected channel for discovery target: " + target);
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

    String selectedTarget() { return discovery.route().master(); }

    ScheduleOutcome schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB request, long timeoutMs) {
        return scheduleDiscovered(request, timeoutMs);
    }

    private ScheduleOutcome scheduleDiscovered(
            FlexlbScheduleProtocol.FlexlbScheduleRequestPB request, long timeoutMs) {
        MasterRouteDiscovery.Route route = discovery.route();
        List<String> targets = new ArrayList<>();
        if (route.master() != null) {
            targets.add(route.master());
        }
        if (route.slave() != null && !route.slave().equals(route.master())) {
            targets.add(route.slave());
        }
        // A node that left the healthy route must not keep its old gRPC
        // channel's reconnect backoff when it becomes healthy again. The
        // frontend closes a target channel on RPC failure and creates a fresh
        // one for the next call; retire our pooled channels at the same edge.
        for (Map.Entry<String, TargetPool> entry : discoveredPools.entrySet()) {
            if (!targets.contains(entry.getKey())) {
                retireDiscoveredPool(entry.getKey(), entry.getValue());
            }
        }
        if (route.master() == null) {
            return new ScheduleOutcome(null, "", false, ErrorKind.TRANSPORT, null, List.of());
        }
        List<ScheduleAttempt> attempts = new ArrayList<>();
        Exception lastFailure = null;
        for (String target : targets) {
            TargetPool pool;
            FlexlbServiceGrpc.FlexlbServiceBlockingStub stub;
            try {
                pool = discoveredPools.computeIfAbsent(target, this::newTargetPool);
                stub = pool.nextStub().withDeadlineAfter(timeoutMs, TimeUnit.MILLISECONDS);
            } catch (RuntimeException failure) {
                return new ScheduleOutcome(null, target, attempts.size() > 1,
                        ErrorKind.BUSINESS, failure, attempts);
            }
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
                if (code != Status.Code.UNAVAILABLE) {
                    return new ScheduleOutcome(null, target, attempts.size() > 1,
                            code == Status.Code.DEADLINE_EXCEEDED ? ErrorKind.DEADLINE : ErrorKind.BUSINESS,
                            e, attempts);
                }
                // A request failure does not own other requests on the shared pool.
                lastFailure = e;
            } catch (RuntimeException e) {
                attempts.add(new ScheduleAttempt(target, startedEpochMs, startedNanos,
                        "EXCEPTION", null));
                return new ScheduleOutcome(null, target, attempts.size() > 1,
                        ErrorKind.BUSINESS, e, attempts);
            }
        }
        return new ScheduleOutcome(null, targets.get(targets.size() - 1), attempts.size() > 1,
                ErrorKind.TRANSPORT, lastFailure, attempts);
    }

    private void retireDiscoveredPool(String target, TargetPool pool) {
        // Injected stubs have no channels. Keep them available across test
        // calls; production pools are removed atomically and closed gently so
        // already-running calls can finish.
        if (pool.channels != null && discoveredPools.remove(target, pool)) {
            for (ManagedChannel channel : pool.channels) {
                channel.shutdown();
            }
        }
    }

    /** Shuts down every target pool's channels (no-op for test-constructed pools). */
    void shutdown() {
        discovery.close();
        for (TargetPool pool : discoveredPools.values()) {
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
