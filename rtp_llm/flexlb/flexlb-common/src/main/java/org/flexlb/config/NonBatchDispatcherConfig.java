package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

/** Controls frontend route delivery, not decision-group formation. */
@Getter
@Setter
public final class NonBatchDispatcherConfig implements DispatcherConfig {

    private Integer maxInflightRequestsPerPrefillWorker;
}
