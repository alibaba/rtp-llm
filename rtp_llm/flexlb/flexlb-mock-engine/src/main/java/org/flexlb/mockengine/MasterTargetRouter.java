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
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Sticky multi-target failover router for the Schedule RPC — the HA case-test
 * counterpart of the legacy single-address path in {@link JavaLoadClient}.
 *
 * <p>Simplified model (deliberately NOT a 1:1 port of the production Python
 * frontend's vipserver discovery layer): the client holds a static list of
 * flexlb gRPC addresses ({@code GRPC_TARGETS=A,B}), sticks to one "current"
 * target, and only reacts to <b>transport-layer</b> failures. The production
 * pull-mode equivalent — polling {@code master/info} every second for
 * {@code real_master_host} — is not implemented; the two are semantically
 * equivalent for the master-side HA contract because a request that lands on
 * the backup is forwarded by it (hop=1) and stays transparent to the client.
 *
 * <p>Per-request decision flow (brief p1 decision tree — event ① "A
 * unreachable" is decoupled from event ② "B became master"; the client never
 * waits for or probes ZK state):
 * <ol>
 *   <li>Send Schedule to the sticky target (per-target channel pool,
 *       round-robin within the pool — same pool shape as the legacy path).</li>
 *   <li>On gRPC {@code UNAVAILABLE} (connection refused/broken — the ONLY
 *       retrying error family): same-request retry on the next target in the
 *       list, wrap-around, one attempt per target. On success from a
 *       different target the sticky pointer moves there (a backup answering
 *       is good enough — whether it is already master is event ②, none of
 *       the client's business).</li>
 *   <li>Every other outcome is terminal for the request:
 *       {@code DEADLINE_EXCEEDED} → no retry/switch/direct-fallback
 *       (single-assertion caliber); any other gRPC status or non-gRPC
 *       exception → treated as "master answered at the gRPC layer" — no
 *       retry, no switch, no direct fallback (same boundary as business
 *       error codes such as 8431/8511).</li>
 *   <li>All targets {@code UNAVAILABLE} → double-connection-failure outcome
 *       ({@link ErrorKind#TRANSPORT}); the caller decides whether the
 *       ENABLE_FALLBACK direct-to-engine escape hatch applies.</li>
 *   <li>Failback is symmetric wrap-around: when the sticky target dies and a
 *       previously-dead target has recovered, the retry chain naturally lands
 *       back on it and the sticky pointer follows — no probing thread, no
 *       direction preference.</li>
 * </ol>
 *
 * <p>Error-code boundary (brief p1 "edge notes" — none of these enter the
 * retry/switch/fallback logic): business codes in the Schedule response
 * (8431 admission rejection, 8511 forwarding terminal code from the
 * split-brain window, ...) are returned to the caller as ordinary responses;
 * the router does not even look at them. Classification of those rows into
 * {@code error_kind=business} happens in the caller.
 *
 * <p>Sticky-pointer concurrency: the pointer is a plain volatile write on
 * success. Concurrent requests during a failover window may briefly diverge
 * before converging on the newly-proven target; this is best-effort by
 * design (the simplified model has no probing, so the next request simply
 * retries). Observability (per-request {@code master_target}/{@code failover}
 * fields) is stamped per request and stays exact regardless of the race.
 */
final class MasterTargetRouter {

    /** Coarse failure taxonomy for the per-request {@code error_kind} field. */
    enum ErrorKind {
        /** Schedule RPC returned a response (any code — business codes are
         *  classified by the caller, not the router). */
        NONE("none"),
        /** Every target failed with gRPC UNAVAILABLE (transport layer). */
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

        ScheduleOutcome(FlexlbScheduleProtocol.FlexlbScheduleResponsePB response,
                String lastTarget, boolean failover, ErrorKind errorKind, Exception failure) {
            this.response = response;
            this.lastTarget = lastTarget;
            this.failover = failover;
            this.errorKind = errorKind;
            this.failure = failure;
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
    /** Index into {@link #pools} of the current sticky target. */
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
            pools.add(new TargetPool(target, channels, stubs));
        }
        this.sticky = 0;
    }

    /**
     * Test constructor: injects stub arrays directly (channels may be null —
     * {@link #shutdown()} skips them).
     */
    MasterTargetRouter(List<String> targets, List<FlexlbServiceGrpc.FlexlbServiceBlockingStub[]> stubs) {
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
    ScheduleOutcome schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB request, long timeoutMs) {
        int start = sticky;
        Exception lastFailure = null;
        for (int attempt = 0; attempt < pools.size(); attempt++) {
            int idx = Math.floorMod(start + attempt, pools.size());
            TargetPool pool = pools.get(idx);
            FlexlbServiceGrpc.FlexlbServiceBlockingStub stub = pool.nextStub()
                    .withDeadlineAfter(timeoutMs, TimeUnit.MILLISECONDS);
            try {
                FlexlbScheduleProtocol.FlexlbScheduleResponsePB response = stub.schedule(request);
                // Any response (code 200 or business error) proves the target
                // alive: the sticky pointer follows it. A business error from
                // the sticky target itself leaves the pointer untouched in
                // practice (idx == sticky) — it only moves when transport
                // failover delivered us to another target first.
                sticky = idx;
                return new ScheduleOutcome(response, pool.target, attempt > 0, ErrorKind.NONE, null);
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
                ErrorKind kind = code == Status.Code.DEADLINE_EXCEEDED
                        ? ErrorKind.DEADLINE : ErrorKind.BUSINESS;
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
        return new ScheduleOutcome(null, pools.get(lastIdx).target, pools.size() > 1,
                ErrorKind.TRANSPORT, lastFailure);
    }

    /** Shuts down every target pool's channels (no-op for test-constructed pools). */
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
}
