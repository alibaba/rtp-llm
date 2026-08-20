package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public final class NonBatchDispatcherConfig implements DispatcherConfig {

    private Integer maxInflightRequestsPerPrefillWorker;
}
