package org.flexlb.mockengine;

import io.grpc.ManagedChannel;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.netty.NettyChannelBuilder;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.socket.nio.NioSocketChannel;
import org.flexlb.api.protocol.FlexlbScheduleProtocol;
import org.flexlb.api.protocol.FlexlbServiceGrpc;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Sends Schedule calls to a sticky master target and tries each remaining target only after {@code
 * UNAVAILABLE}. Success updates the sticky target; other transport statuses and business responses
 * finish the attempt, while exhaustion leaves direct fallback to the caller.
 */
final class MasterTargetRouter {
    private final List<TargetPool> pools;

    /**
     * Index into {@link #pools} of the current sticky target.
     */
    private volatile int sticky;

    /**
     * Production constructor: builds an N_CHANNELS channel pool per target,
     * with the same channel options as the legacy single-target path in
     * JavaLoadClient.
     */
    MasterTargetRouter(List<String> targets, int nChannels, EventLoopGroup eventLoopGroup) {
        this.pools = new ArrayList<>(targets.size());
        for (String target : targets) {
            ManagedChannel[] channels = new ManagedChannel[nChannels];
            FlexlbServiceGrpc.FlexlbServiceBlockingStub[] stubs =
                    new FlexlbServiceGrpc.FlexlbServiceBlockingStub[nChannels];
            for (int i = 0; i < nChannels; i++) {
                ManagedChannel channel =
                        NettyChannelBuilder.forTarget(target)
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
            pools.add(new TargetPool(target, channels, stubs));
        }
        this.sticky = 0;
    }

    /**
     * Test constructor: injects stub arrays directly (channels may be null —
     * {@link #shutdown()} skips them).
     */
    MasterTargetRouter(
            List<String> targets, List<FlexlbServiceGrpc.FlexlbServiceBlockingStub[]> stubs) {
        this.pools = new ArrayList<>(targets.size());
        for (int i = 0; i < targets.size(); i++) {
            pools.add(new TargetPool(targets.get(i), null, stubs.get(i)));
        }
        this.sticky = 0;
    }

    String stickyTarget() {
        return pools.get(sticky).target;
    }

    /**
     * One failover-aware Schedule call. See the class javadoc for the decision
     * flow; the method never throws — every failure is folded into the
     * returned {@link ScheduleOutcome}.
     */
    ScheduleOutcome schedule(
            FlexlbScheduleProtocol.FlexlbScheduleRequestPB request, long timeoutMs) {
        int start = sticky;
        Exception lastFailure = null;
        for (int attempt = 0; attempt < pools.size(); attempt++) {
            int idx = Math.floorMod(start + attempt, pools.size());
            TargetPool pool = pools.get(idx);
            FlexlbServiceGrpc.FlexlbServiceBlockingStub stub =
                    pool.nextStub().withDeadlineAfter(timeoutMs, TimeUnit.MILLISECONDS);
            try {
                FlexlbScheduleProtocol.FlexlbScheduleResponsePB response = stub.schedule(request);
                // Any response (code 200 or business error) proves the target
                // alive: the sticky pointer follows it. A business error from
                // the sticky target itself leaves the pointer untouched in
                // practice (idx == sticky) — it only moves when transport
                // failover delivered us to another target first.
                sticky = idx;
                return new ScheduleOutcome(
                        response, pool.target, attempt > 0, ErrorKind.NONE, null);
            } catch (StatusRuntimeException e) {
                Status.Code code = e.getStatus().getCode();
                if (code == Status.Code.UNAVAILABLE) {
                    // Event ① (transport-layer unreachable): same-request retry
                    // on the next target — never wait for ZK, never probe.
                    lastFailure = e;
                    continue;
                }
                // DEADLINE_EXCEEDED and every other gRPC status: terminal for
                // this request — no retry, no switch, no direct fallback.
                ErrorKind kind =
                        code == Status.Code.DEADLINE_EXCEEDED
                                ? ErrorKind.DEADLINE
                                : ErrorKind.BUSINESS;
                return new ScheduleOutcome(null, pool.target, attempt > 0, kind, e);
            } catch (RuntimeException e) {
                // Non-gRPC failure: conservative — only UNAVAILABLE retries.
                return new ScheduleOutcome(null, pool.target, attempt > 0, ErrorKind.BUSINESS, e);
            }
        }
        // Every target answered UNAVAILABLE: double connection failure. The
        // sticky pointer stays put — with no probing thread the next request
        // simply repeats the failover chain (observable as failover=true rows
        // while the outage lasts, matching the "fallback <= 1 per request"
        // case-test assertions).
        int lastIdx = Math.floorMod(start + pools.size() - 1, pools.size());
        return new ScheduleOutcome(
                null,
                pools.get(lastIdx).target,
                pools.size() > 1,
                ErrorKind.TRANSPORT,
                lastFailure);
    }

    /**
     * Shuts down every target pool's channels (no-op for test-constructed pools).
     */
    void shutdown() {
        for (TargetPool pool : pools) {
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

    /**
     * Coarse failure taxonomy for the per-request {@code error_kind} field.
     */
    enum ErrorKind {
        /** Schedule RPC returned a response (any code — business codes are
         *  classified by the caller, not the router). */
        NONE("none"),
        /**
         * Every target failed with gRPC UNAVAILABLE (transport layer).
         */
        TRANSPORT("transport"),
        /**
         * gRPC DEADLINE_EXCEEDED — no retry, no switch, no fallback.
         */
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

    /**
     * Terminal outcome of one failover-aware Schedule call.
     */
    static final class ScheduleOutcome {
        /**
         * Master response, or null when no target produced one.
         */
        final FlexlbScheduleProtocol.FlexlbScheduleResponsePB response;

        /**
         * Actually-served (or last-attempted, when none served) target address.
         */
        final String lastTarget;

        /**
         * True when the same request was retried on another target.
         */
        final boolean failover;

        final ErrorKind errorKind;

        /**
         * Transport-layer exception of the last UNAVAILABLE attempt (null otherwise).
         */
        final Exception failure;

        ScheduleOutcome(
                FlexlbScheduleProtocol.FlexlbScheduleResponsePB response,
                String lastTarget,
                boolean failover,
                ErrorKind errorKind,
                Exception failure) {
            this.response = response;
            this.lastTarget = lastTarget;
            this.failover = failover;
            this.errorKind = errorKind;
            this.failure = failure;
        }
    }

    /**
     * One target's channel pool — same N_CHANNELS + round-robin shape as the legacy single-target path.
     */
    private static final class TargetPool {
        final String target;
        final ManagedChannel[] channels;
        final FlexlbServiceGrpc.FlexlbServiceBlockingStub[] stubs;
        final AtomicInteger rr = new AtomicInteger();

        TargetPool(
                String target,
                ManagedChannel[] channels,
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
}
