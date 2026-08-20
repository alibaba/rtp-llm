package org.flexlb.config;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, property = "type")
@JsonSubTypes({
        @JsonSubTypes.Type(value = BatchDispatcherConfig.class, name = "BATCH"),
        @JsonSubTypes.Type(value = NonBatchDispatcherConfig.class, name = "NON_BATCH")
})
public sealed interface DispatcherConfig permits BatchDispatcherConfig, NonBatchDispatcherConfig {
}
