package org.flexlb.balance.endpoint;

/** Independent sink for exact endpoint-owned facts and retirement events. */
@FunctionalInterface
public interface EndpointEventSink {
    void onEndpointEvent(EndpointEvent event);
}
