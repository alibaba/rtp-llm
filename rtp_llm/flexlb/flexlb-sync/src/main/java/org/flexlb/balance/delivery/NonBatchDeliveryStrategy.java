package org.flexlb.balance.delivery;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.PrefillAdmissionResources;
import org.flexlb.balance.scheduler.RequestRegistry;
import org.flexlb.balance.scheduler.RequestSlot.DeliveryClaim;
import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.ArrayList;
import java.util.List;

import static org.flexlb.balance.scheduler.PrefillAdmissionResources.missingEndpoint;

/**
 * NON_BATCH 模式的投递策略：请求逐个投递给 Prefill，不攒批。
 * 负责预留容量、提交归属、发布路由和估算 TTFT。
 */
public final class NonBatchDeliveryStrategy implements DeliveryStrategy {

    private final RequestRegistry requests;
    private final DeliveryMetrics telemetry;

    public NonBatchDeliveryStrategy(RequestRegistry requests, DeliveryMetrics telemetry) {
        this.requests = requests;
        this.telemetry = telemetry;
    }

    @Override
    public DeliveryTransaction prepare(DeliveryGroup group) {
        List<ScheduledRequest> candidates = group.requests();
        PrefillTimePredictor.Evaluator evaluator = group.evaluator();
        ScheduledRequest head = candidates.get(0);
        List<ScheduledRequest> requestsPreparedForDelivery = new ArrayList<>(candidates.size());
        PrefillEndpoint prefill = head.prefillEp();
        if (prefill == null) {
            return NonBatchTransaction.blocked(
                    this,
                    new BlockedRequest(head, DeliveryBlockReason.failed(missingEndpoint("Prefill", head))));
        }
        NonBatchTransaction transaction = new NonBatchTransaction(this, prefill);
        BlockedRequest blocked = null;
        try {
            for (ScheduledRequest item : candidates) {
                if (prefill.isEngineSubQueueMode()) {
                    long maxDispatchLimit = item.ctx().getConfig().getDispatcher().getMaxInflightPerPrefillWorker();
                    if (!prefill.canDeliver(maxDispatchLimit, requestsPreparedForDelivery.size() + 1)) {
                        blocked = new BlockedRequest(item, DeliveryBlockReason.deliveryUnavailable(
                                prefill.routeAdmissionAvailability(maxDispatchLimit)));
                        break;
                    }
                }
                DeliveryBlockReason.Attempt<ScheduledRequest> attempt =
                        requests.prepareRouteMember(item, transaction, evaluator);
                if (!attempt.accepted()) {
                    blocked = new BlockedRequest(item, attempt.blockReason());
                    break;
                }
                requestsPreparedForDelivery.add(item);
            }
            if (requestsPreparedForDelivery.isEmpty()) {
                Throwable cleanup = close(transaction);
                if (cleanup != null) {
                    throw cleanup;
                }
                return NonBatchTransaction.blocked(this, blocked);
            }
            transaction.select(requestsPreparedForDelivery, blocked);
            return transaction;
        } catch (Throwable failure) {
            Throwable cleanup = close(transaction);
            if (cleanup != null && cleanup != failure) {
                failure.addSuppressed(cleanup);
            }
            throw propagate(failure);
        }
    }

    void deliver(NonBatchTransaction transaction,
                 PrefillAdmissionResources.CommittedAdmissionOwner admission,
                 int remainingQueueDepth,
                 WorkSnapshot precedingWork) {
        List<ScheduledRequest> items = transaction.requests();
        Throwable deliveryFailure = null;
        List<ScheduledRequest> delivered = new ArrayList<>(items.size());
        List<ClaimedRoute> claimed = new ArrayList<>(items.size());
        try {
            for (ScheduledRequest item : items) {
                DeliveryClaim claim;
                try {
                    claim = requests.claimRouteDelivery(
                            item, admission);
                } catch (Throwable claimFailure) {
                    try {
                        requests.failDeliveryPreparation(item, claimFailure);
                    } catch (Throwable terminalFailure) {
                        claimFailure.addSuppressed(terminalFailure);
                        deliveryFailure = append(
                                deliveryFailure, claimFailure);
                    }
                    continue;
                }
                if (claim == null) {
                    continue;
                }
                claimed.add(new ClaimedRoute(item, claim));
            }
            long unstartedWorkMs = 0L;
            for (ClaimedRoute route : claimed) {
                ScheduledRequest item = route.item();
                try {
                    long itemWorkMs = transaction.predictedMs(item);
                    unstartedWorkMs = unstartedWorkMs > Long.MAX_VALUE - itemWorkMs
                            ? Long.MAX_VALUE : unstartedWorkMs + itemWorkMs;
                    requests.publishRoute(route.claim(), precedingWork, unstartedWorkMs);
                    delivered.add(item);
                } catch (Throwable completionFailure) {
                    deliveryFailure = append(
                            deliveryFailure, completionFailure);
                }
            }
        } finally {
            deliveryFailure = append(
                    deliveryFailure, close(admission));
        }
        if (!delivered.isEmpty()) {
            telemetry.routesDelivered(remainingQueueDepth, delivered);
        }
        if (deliveryFailure != null) {
            throw propagate(deliveryFailure);
        }
    }

    private record ClaimedRoute(ScheduledRequest item, DeliveryClaim claim) { }

    static long predict(PrefillTimePredictor.Evaluator evaluator, ScheduledRequest item) {
        return PrefillPredictionBoundary.predictSingleRequestMs(
                evaluator, item.seqLen(), item.hitCache());
    }

    static Throwable close(AutoCloseable capability) {
        try {
            capability.close();
            return null;
        } catch (Throwable failure) {
            return failure;
        }
    }

    private static Throwable append(Throwable first, Throwable next) {
        if (next == null) {
            return first;
        }
        if (first == null) {
            return next;
        }
        if (first != next) {
            first.addSuppressed(next);
        }
        return first;
    }

    static RuntimeException propagate(Throwable failure) {
        if (failure instanceof RuntimeException runtimeFailure) {
            return runtimeFailure;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        return new IllegalStateException("route delivery failed", failure);
    }
}
