package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryLifecyclePort;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillGenerationRuntime;
import org.flexlb.config.FlexlbConfig;
import org.springframework.stereotype.Component;

/** Composition-owned constructor for per-endpoint Prefill runtimes. */
@Component
final class WorkerBatcherFactory
        implements PrefillGenerationRuntime.Factory {

    @Override
    public PrefillGenerationRuntime.Generation create(
            String endpointId,
            PrefillEndpoint endpoint,
            FlexlbConfig config,
            DeliveryStrategy deliveryStrategy,
            DeliveryLifecyclePort deliveryLifecycle) {
        // Construction containment: WorkerBatcher and BatcherContext only
        // retain the exact endpoint. They never dereference it, start a thread,
        // or publish a callback during this call.
        WorkerBatcher runtime = new WorkerBatcher(
                endpointId,
                endpoint,
                config,
                deliveryStrategy,
                deliveryLifecycle);
        return new PrefillGenerationRuntime.Generation(
                runtime, runtime.ownedLedger());
    }
}
