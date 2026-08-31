package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EndpointEventSink;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.springframework.stereotype.Component;

/** Composition-owned constructor for per-endpoint Prefill runtimes. */
@Component
public final class WorkerBatcherFactory {

    public WorkerBatcherFactory() {
    }

    public WorkerBatcher create(
            String endpointId,
            PrefillEndpoint endpoint,
            FlexlbConfig config,
            DeliveryStrategy deliveryStrategy,
            EndpointEventSink deliveryLifecycle) {
        // Construction containment: WorkerBatcher and BatcherContext only
        // retain the exact endpoint. They never dereference it, start a thread,
        // or publish a callback during this call.
        WorkerBatcher runtime = new WorkerBatcher(
                endpointId,
                endpoint,
                config,
                deliveryStrategy,
                deliveryLifecycle);
        return runtime;
    }
}
