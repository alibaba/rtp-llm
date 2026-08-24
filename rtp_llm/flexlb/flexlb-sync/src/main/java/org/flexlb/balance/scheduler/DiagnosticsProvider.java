package org.flexlb.balance.scheduler;

import java.util.Map;

/**
 * Generic diagnostics interface for components that can report internal
 * state to the HTTP diagnostic endpoints ({@code /inflight_status},
 * {@code /rtp_llm/master/info}).
 *
 * <p>Implemented by {@link AbstractScheduler} (schedulers) and
 * {@code EndpointRegistry} (EP counts). {@code RouteService} collects all
 * providers and exposes them via
 * {@code getDiagnosticsProviders()} so that {@code HttpLoadBalanceServer}
 * can aggregate diagnostics without hard-coded QUEUE-specific method calls
 * on individual components.
 *
 * <p>The returned map keys are free-form strings; callers search for
 * well-known keys ({@code "queue_length"}, {@code "active_count"},
 * {@code "queue_snapshot"}, etc.). An empty map means the component has
 * no diagnostics to report.
 */
public interface DiagnosticsProvider {

    /**
     * Return a snapshot of this component's internal diagnostics.
     *
     * @return a map of diagnostic key → value; empty if nothing to report
     */
    Map<String, Object> getDiagnostics();
}
