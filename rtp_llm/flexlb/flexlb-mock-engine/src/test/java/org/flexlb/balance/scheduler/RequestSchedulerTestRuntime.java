package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.priority.DecodePreemptionCoordinator;
import org.flexlb.balance.scheduler.priority.EngineCancelChannel;
import org.flexlb.balance.scheduler.priority.PlanCommitter;
import org.flexlb.balance.scheduler.priority.PriorityAdmissionScheduler;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.PrioritySchedulerReporter;

import java.util.Objects;
import java.util.function.Function;

/**
 * Test-only composition root assembling the dsv4 (schema-v1) scheduler stack:
 * {@link EndpointRegistry} + {@link PriorityAdmissionScheduler} +
 * {@link PriorityScheduler} + a {@link BatchDispatcher}, mirroring the
 * production bean wiring for in-process end-to-end tests.
 *
 * <p>This is the old-stack counterpart of the intake3
 * {@code RequestSchedulerTestRuntime}: instead of the v2
 * RequestRegistry/EvictionManager/BatchDeliveryStrategy composition it wires
 * the legacy PriorityScheduler family, including the
 * {@code selectPrefillForDecodeEviction} hook the admission scheduler needs
 * when a decode-eviction victim must be re-routed to a prefill worker.
 */
public final class RequestSchedulerTestRuntime implements AutoCloseable {

    private final PriorityAdmissionScheduler admissionScheduler;
    private final PriorityScheduler scheduler;
    private final BatchDispatcher batchDispatcher;

    /**
     * @param evictionPrefillSelector decode-eviction re-route hook: maps the
     *                                incoming context to the prefill
     *                                {@link ServerStatus} the victim should be
     *                                handed off to (may return null to fail the
     *                                re-route).
     */
    public RequestSchedulerTestRuntime(
            ConfigService configService,
            Router router,
            EndpointRegistry endpointRegistry,
            BatchDispatcher batchDispatcher,
            BatchSchedulerReporter batchReporter,
            PrioritySchedulerReporter priorityReporter,
            EngineCancelChannel cancelChannel,
            Function<BalanceContext, ServerStatus> evictionPrefillSelector) {
        Objects.requireNonNull(endpointRegistry, "endpointRegistry");
        Objects.requireNonNull(batchDispatcher, "batchDispatcher");
        Objects.requireNonNull(cancelChannel, "cancelChannel");
        Objects.requireNonNull(evictionPrefillSelector, "evictionPrefillSelector");

        this.batchDispatcher = batchDispatcher;
        this.admissionScheduler = new PriorityAdmissionScheduler(
                configService, router, endpointRegistry, new PlanCommitter(),
                priorityReporter, batchReporter, cancelChannel,
                new DecodePreemptionCoordinator(cancelChannel)) {
            @Override
            protected ServerStatus selectPrefillForDecodeEviction(
                    BalanceContext ctx, org.flexlb.config.FlexlbConfig config,
                    String group) {
                return evictionPrefillSelector.apply(ctx);
            }
        };
        this.scheduler = new PriorityScheduler(
                configService, router, endpointRegistry, batchDispatcher,
                batchReporter, admissionScheduler, null, cancelChannel);
    }

    public PriorityScheduler scheduler() {
        return scheduler;
    }

    public PriorityAdmissionScheduler admissionScheduler() {
        return admissionScheduler;
    }

    @Override
    public void close() {
        scheduler.shutdown();
    }
}
