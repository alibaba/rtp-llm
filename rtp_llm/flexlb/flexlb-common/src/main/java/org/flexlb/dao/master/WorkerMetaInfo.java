package org.flexlb.dao.master;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@JsonIgnoreProperties(ignoreUnknown = true)
@Data
@AllArgsConstructor
@NoArgsConstructor
public class WorkerMetaInfo {

    @JsonProperty
    private String ip;

    @JsonProperty
    private int port;

    @JsonProperty
    private String site;

    @JsonProperty
    private int availableConcurrency;

    @JsonProperty
    private int currentConcurrency;

    @JsonProperty
    private int maxConcurrency;

    @JsonProperty
    private String errorMsg;

    @JsonProperty
    private String gpu;
}
