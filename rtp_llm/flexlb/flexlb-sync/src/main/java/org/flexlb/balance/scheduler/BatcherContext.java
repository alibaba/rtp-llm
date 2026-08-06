package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.service.monitor.BatchSchedulerReporter;

/**
 * Execution context for {@link WorkerBatcher}: holds the prefill endpoint
 * reference, runtime config, and metric reporter needed by the batcher's
 * {@code executeDispatch} / {@code executeDrop} side-effect methods.
 *
 * <p>Queue management has been moved entirely into
 * {@link FixedWindowBatcherAlgorithm}; this class no longer holds or
 * exposes any queue state.
 */
public class BatcherContext {

    private final String key;
    private final PrefillEndpoint prefillEp;
    private final FlexlbConfig cfg;
    private final BatchSchedulerReporter reporter;

    BatcherContext(String key, PrefillEndpoint prefillEp, FlexlbConfig cfg,
                   BatchSchedulerReporter reporter) {
        this.key = key;
        this.prefillEp = prefillEp;
        this.cfg = cfg;
        this.reporter = reporter;
    }

    // ---- accessors ----

    String key() {
        return key;
    }

    PrefillEndpoint prefillEp() {
        return prefillEp;
    }

    FlexlbConfig cfg() {
        return cfg;
    }

    BatchSchedulerReporter reporter() {
        return reporter;
    }
}
