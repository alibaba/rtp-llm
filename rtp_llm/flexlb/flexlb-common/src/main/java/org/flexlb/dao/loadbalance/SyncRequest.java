package org.flexlb.dao.loadbalance;

import com.alibaba.fastjson.JSONArray;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import org.flexlb.enums.LoadBalanceStrategyEnum;


@Data
public class SyncRequest {

    @JsonProperty("model")
    private String model;

    @JsonProperty("chat_id")
    private String chatId;

    @JsonProperty("historyBackendAddress")
    private JSONArray historyBackendAddress;

    @JsonProperty("backendWorkerSite")
    private String backendWorkerSite;

    @JsonProperty("loadBalanceStrategy")
    private LoadBalanceStrategyEnum loadBalanceStrategy;
}