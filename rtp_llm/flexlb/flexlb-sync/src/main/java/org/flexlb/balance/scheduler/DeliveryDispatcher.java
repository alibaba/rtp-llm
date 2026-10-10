package org.flexlb.balance.scheduler;

import io.micrometer.core.instrument.MeterRegistry;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.telemetry.FlexlbTrace;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

/** Complete payload preparation precedes sending; Scheduler owns all request transitions. */
@Component
public class DeliveryDispatcher {
    static final long CLEANUP_TIMEOUT_MS = DeliveryCleanup.TIMEOUT_MS;
    private final DeliveryCleanup cleanup = new DeliveryCleanup();
    private final EngineGrpcClient grpcClient;
    private final DispatcherConfig.Type mode;
    private final ConfigService configService;
    private final DeliveryMetricsReporter telemetry;

    @Autowired
    public DeliveryDispatcher(EngineGrpcClient grpcClient, ConfigService configService,
            @Autowired(required = false) MeterRegistry registry, DeliveryMetricsReporter telemetry, Environment environment) {
        this.grpcClient = grpcClient;
        this.configService = Objects.requireNonNull(configService);
        this.mode = configService.loadBalanceConfig().getDispatcher().getType();
        this.telemetry = Objects.requireNonNull(telemetry);
    }

    public DispatcherConfig.Type mode() { return mode; }

    record Member(RequestRoute route, DecodeEndpoint.EngineDispatchPermit decode) implements AutoCloseable {
        private static final RouteProjection.AdmissionBlockSemantics DECODE_BLOCK =
                new RouteProjection.AdmissionBlockSemantics("DELIVERY_CAPACITY_DECODE_ENGINE",
                        RouteProjection.AfterProbeAdmission.UNAVAILABLE, "DECODE_CAPACITY_SCOPE_UNKNOWN", RoleType.DECODE);

        static CapacityBoundary.Attempt<Member> prepare(RequestRoute route, AdmissionCapacity capacity) {
            var endpoint = route.decodeEp();
            var reservation = route.decodeReservation();
            if (endpoint == null && reservation == null) { return CapacityBoundary.Attempt.accepted(new Member(route, null)); }
            if (endpoint == null || reservation == null) {
                return CapacityBoundary.Attempt.rejected(CapacityBoundary.failed(new IllegalStateException(
                        "Decode reservation is unavailable: request_id=" + route.requestId())));
            }
            DecodeEndpoint.EngineDispatchPermit permit = null;
            try {
                var acquisition = endpoint.acquireDispatchPermit(reservation, capacity);
                switch (acquisition.status()) {
                    case ACQUIRED, ALREADY_ACCEPTED -> permit = acquisition.handle();
                    case CAPACITY_FULL -> { return CapacityBoundary.Attempt.rejected(CapacityBoundary.unavailable(
                            CapacityBoundary.Resource.DECODE_ENGINE, DECODE_BLOCK)); }
                    case NOT_OWNED, NOT_QUEUED -> { return CapacityBoundary.Attempt.rejected(CapacityBoundary.OWNERSHIP_LOST); }
                    case ENDPOINT_RETIRED -> { return CapacityBoundary.Attempt.rejected(CapacityBoundary.failed(
                            new IllegalStateException("Decode endpoint generation retired: request_id=" + route.requestId()))); }
                    case ALREADY_ACQUIRED -> { return CapacityBoundary.Attempt.rejected(CapacityBoundary.failed(
                            new IllegalStateException("Decode dispatch permit already acquired: request_id=" + route.requestId()))); }
                }
                return CapacityBoundary.Attempt.accepted(new Member(route, permit));
            } catch (Throwable failure) {
                if (permit != null) { failure = Failures.run(failure, permit::release); }
                return CapacityBoundary.Attempt.rejected(CapacityBoundary.failed(failure));
            }
        }

        @Override public void close() { if (decode != null) { decode.release(); } }
    }

    /** Entering deliverBatch transfers all temporary resources, including on rejection. */
    record Batch(long batchId, List<Member> members, WorkSnapshot precedingWork, long predictedMs,
            PrefillTimePredictor.Evaluator evaluator, String reason, int remainingQueueDepth,
            PrefillState.CommittedHandoff handoff, DeliveryExecution.DeliveryPermit executionPermit) {
        Batch {
            members = List.copyOf(members);
            checkArgument(!members.isEmpty() && batchId > 0 && predictedMs >= 0 && remainingQueueDepth >= 0, "invalid batch metadata");
            Objects.requireNonNull(precedingWork);
            Objects.requireNonNull(evaluator);
            Objects.requireNonNull(reason);
            Objects.requireNonNull(executionPermit);
        }
    }

    private record BatchCall(List<DeliveryClaim> pending, List<DeliveryClaim> skipped,
            CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> response, DeliveryResult failure) { }

    void deliverBatch(Batch batch) {
        batch.executionPermit().submit(() -> deliverCommitted(batch), failure -> failBeforeSend(batch, List.of(), failure));
    }

    private void deliverCommitted(Batch batch) {
        var claims = new ArrayList<DeliveryClaim>(batch.members().size());
        long prediction = batch.predictedMs();
        try {
            claim(batch, claims);
            if (claims.size() != batch.members().size() && !claims.isEmpty()) {
                prediction = PrefillPredictionBoundary.predictCommittedBatchMs(batch.evaluator(),
                        PrefillBatchFeatures.from(claims, value -> value.item().seqLen(), value -> value.item().hitCache()));
            }
            for (DeliveryClaim claim : claims) {
                long now = System.currentTimeMillis();
                claim.owner().data().getScheduler().onDeliveryEvidence(claim, new ContextResult.Prediction(claim,
                        PrefillPredictionBoundary.predictedVisibilityDeadlineMs(batch.precedingWork(), prediction,
                                claim.owner().data().getConfig().getRequestLifecycle().getDecision().getLifetime(), now), now));
            }
        } catch (Throwable failure) { failBeforeSend(batch, claims, failure); return; }
        BatchCall call;
        try { call = dispatchBatch(batch, claims, prediction); }
        finally {
            Throwable cleanup = closeCommitted(batch.members(), batch.handoff());
            if (cleanup != null) { batch.members().getFirst().route().ctx().data().getScheduler().runtime.recordFailure(cleanup); }
        }
        if (call.response() == null) { batch.executionPermit().releaseIfNoResponse(); }
        report(batch.batchId(), call.skipped(), Collections.nCopies(call.skipped().size(),
                DeliveryResult.notSent(new CancellationException("delivery abandoned before send"))));
        if (call.failure() != null) {
            report(batch.batchId(), call.pending(), Collections.nCopies(call.pending().size(), call.failure()));
        } else if (call.response() != null) {
            batch.executionPermit().observeResponse(call.response(), (response, failure) -> report(batch.batchId(), call.pending(),
                    BatchCodec.validate(batch.batchId(), call.pending(), response, failure)));
        }
        if (!claims.isEmpty()) {
            try { telemetry.reportDelivery(batch.batchId(), batch.reason(), batch.remainingQueueDepth(),
                    claims.stream().map(value -> value.item()).toList(), prediction); }
            catch (Throwable failure) { logFailure("Delivery telemetry", batch.batchId(), failure); }
        }
    }

    private void claim(Batch batch, List<DeliveryClaim> claims) {
        for (Member member : batch.members()) {
            try {
                var claim = member.route().ctx().data().getScheduler().claimDelivery(member.route(),
                        DeliveryClaimKind.BATCH_ENQUEUE, batch.batchId(), member.decode());
                if (claim != null) { claims.add(claim); }
            } catch (Throwable failure) {
                try { member.route().ctx().data().getScheduler().onDeliveryEvidence(null,
                        new ContextResult.PreparationFailure(null, member.route(), failure,
                                String.valueOf(failure.getMessage()), System.currentTimeMillis())); }
                catch (Throwable ownerFailure) {
                    member.route().ctx().data().getScheduler().runtime.recordFailure(ownerFailure);
                    logFailure("Claim failure notification", batch.batchId(), ownerFailure);
                }
            }
        }
    }

    private BatchCall dispatchBatch(Batch batch, List<DeliveryClaim> claims, long prediction) {
        var skipped = new ArrayList<DeliveryClaim>();
        var sending = new ArrayList<DeliveryClaim>(claims.size());
        if (claims.isEmpty()) { return new BatchCall(List.of(), List.of(), null, null); }
        boolean invoked = false;
        try {
            long payloadBytes = 16L;
            for (var claim : claims) { payloadBytes += claim.item().batchPayloadSizeUpperBound(); }
            checkState(batch.executionPermit().tryChargeBytes(payloadBytes), "in-flight payload byte budget exhausted");
            var request = BatchCodec.prepare(batch.batchId(), claims);
            checkState(configService.loadBalanceConfig().getDispatcher().getType() == DispatcherConfig.Type.BATCH,
                    "batch submission requires BATCH dispatcher configuration");
            for (DeliveryClaim claim : claims) {
                if (claim.owner().data().getScheduler().tryStartSend(claim)) { sending.add(claim); }
                else { skipped.add(claim); }
            }
            if (sending.isEmpty()) { return new BatchCall(List.of(), List.copyOf(skipped), null, null); }
            if (sending.size() != claims.size()) { request = BatchCodec.retain(request, sending); }
            var items = sending.stream().map(value -> value.item()).toList();
            var endpoint = items.getFirst().prefillEp();
            try { logDispatch(batch.batchId(), items, endpoint, prediction, batch.reason()); }
            catch (Throwable failure) { logFailure("Dispatch logging", batch.batchId(), failure); }
            long now = System.nanoTime();
            for (var item : items) {
                item.ctx().data().setBatchDispatchedNanos(now);
                FlexlbTrace.setScheduleAttribute(item.ctx().data().getTraceContext(), FlexlbTrace.BATCH_ID, batch.batchId());
                FlexlbTrace.setScheduleAttribute(item.ctx().data().getTraceContext(), FlexlbTrace.BATCH_SIZE, (long) items.size());
                FlexlbTrace.setScheduleAttribute(item.ctx().data().getTraceContext(), FlexlbTrace.DISPATCH_REASON, batch.reason());
            }
            String ip = endpoint.getIp();
            int port = endpoint.getGrpcPort();
            var pending = List.copyOf(sending);
            invoked = true;
            var response = Objects.requireNonNull(grpcClient.batchEnqueueAsync(ip, port, request),
                    "EnqueueBatch client returned null future after invocation");
            return new BatchCall(pending, List.copyOf(skipped), response, null);
        } catch (Throwable failure) {
            // Before invocation even already-begun members were not sent; afterwards delivery is uncertain.
            var unresolved = new ArrayList<DeliveryClaim>(claims);
            unresolved.removeAll(skipped);
            logFailure("Dispatch batch", batch.batchId(), failure);
            return new BatchCall(List.copyOf(unresolved), List.copyOf(skipped), null,
                    invoked ? DeliveryResult.uncertain(failure) : DeliveryResult.notSent(failure));
        }
    }

    private static void report(long batchId, List<DeliveryClaim> claims, List<DeliveryResult> results) {
        long responseNanos = System.nanoTime();
        for (int i = 0; i < claims.size(); i++) {
            var claim = claims.get(i);
            var result = results.get(i);
            if (result.status() == DeliveryResult.Status.DELIVERED || result.status() == DeliveryResult.Status.PREFILL_REJECTED) {
                try { FlexlbTrace.setScheduleDuration(claim.owner().data().getTraceContext(), FlexlbTrace.ENQUEUE_BATCH_MS,
                        claim.owner().data().getBatchDispatchedNanos(), responseNanos); }
                catch (Throwable failure) { logFailure("Response trace", batchId, failure); }
            }
            try { claim.owner().data().getScheduler().onDeliveryEvidence(claim,
                    new ContextResult.TransportResult(claim, result, System.currentTimeMillis())); }
            catch (Throwable failure) {
                claim.owner().data().getScheduler().runtime.recordFailure(failure);
                logFailure("Delivery completion request_id=" + claim.item().requestId(), batchId, failure);
            }
        }
    }

    private void failBeforeSend(Batch batch, List<DeliveryClaim> claims, Throwable failure) {
        batch.executionPermit().releaseIfNoResponse();
        try {
            for (Member member : batch.members()) {
                if (claims.stream().anyMatch(value -> value.item() == member.route())) { continue; }
                try { member.route().ctx().data().getScheduler().onDeliveryEvidence(null,
                        new ContextResult.PreparationFailure(null, member.route(), failure,
                                String.valueOf(failure.getMessage()), System.currentTimeMillis())); }
                catch (Throwable ownerFailure) {
                    member.route().ctx().data().getScheduler().runtime.recordFailure(ownerFailure);
                    logFailure("Preparation failure notification", batch.batchId(), ownerFailure);
                }
            }
        } finally {
            Throwable cleanup = closeCommitted(batch.members(), batch.handoff());
            if (cleanup != null) { batch.members().getFirst().route().ctx().data().getScheduler().runtime.recordFailure(cleanup); }
        }
        report(batch.batchId(), claims, Collections.nCopies(claims.size(), DeliveryResult.notSent(failure)));
    }

    static Throwable closeCommitted(List<Member> members, PrefillState.CommittedHandoff handoff) {
        Throwable failure = null;
        for (Member member : members) {
            failure = Failures.append(failure, Failures.close(member));
        }
        return Failures.append(failure, handoff == null ? null : Failures.close(handoff));
    }

    private static void logFailure(String operation, long batchId, Throwable failure) {
        try { Logger.error("{} batch_id={}", operation, batchId, failure); }
        catch (Throwable ignored) { }
    }

    /** Deduplication ends only at physical settlement, independently of the bounded transport session. */
    CompletableFuture<Void> cleanupDelivery(DeliveryClaim claim, EngineCancelChannel channel, SchedulerRuntime runtime) {
        return cleanup.start(claim, channel, runtime);
    }

    private void logDispatch(long batchId, List<RequestRoute> items,
                             PrefillEndpoint prefillEp, long predMs, String reason) {
        if (!Logger.isDebugEnabled()) {
            return;
        }
        long totalTokens = 0;
        long totalHit = 0;
        StringBuilder itemDetail = new StringBuilder();
        for (int i = 0; i < items.size(); i++) {
            RequestRoute item = items.get(i);
            long seqLen = item.seqLen();
            long hitCache = item.hitCache();
            totalTokens += seqLen;
            totalHit += hitCache;
            if (i > 0) {
                itemDetail.append(", ");
            }
            itemDetail.append("{req_id=").append(item.requestId())
                    .append(" seq_len=").append(seqLen)
                    .append(" hit_cache=").append(hitCache).append('}');
        }

        RequestRoute head = items.get(0);
        long now = System.currentTimeMillis();
        long waitMs = now - head.enqueuedAtMs();
        long remainingMs = head.expiresAtMs() - now;

        Logger.debug("flexlb_batch_dispatch batch_id={} batch_size={} total_tokens={} total_hit={} "
                        + "pred_ms={} reason={} wait_ms={} request_remaining_ms={} "
                        + "prefill={}:{} items=[{}]",
                batchId, items.size(), totalTokens, totalHit, predMs, reason,
                waitMs, remainingMs,
                prefillEp.getIp(), prefillEp.getHttpPort(),
                itemDetail);
    }

}
