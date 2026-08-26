package org.flexlb.config;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, property = "type")
@JsonSubTypes({
        @JsonSubTypes.Type(value = BatchDispatcherConfig.class, name = "BATCH"),
        @JsonSubTypes.Type(value = NonBatchDispatcherConfig.class, name = "NON_BATCH")
})
public sealed interface DispatcherConfig permits BatchDispatcherConfig, NonBatchDispatcherConfig {

    /** Stable discriminator used in startup diagnostics. */
    String typeName();

    /** Whether this delivery transport requires the serialized Engine request. */
    boolean requiresGenerateInput();

    /**
     * Per-Prefill-worker inflight limit in the active delivery mode's unit.
     * {@code null} means the optional limit is disabled.
     */
    Integer maxInflightDeliveriesPerPrefillWorker();

    /** Validate mode-owned fields and scheduler compatibility. */
    void validateFor(SchedulerConfig scheduler);
}
