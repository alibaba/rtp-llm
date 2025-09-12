package org.flexlb.dao.master;

import lombok.Data;
import org.flexlb.dao.loadbalance.SyncRequest;
import org.flexlb.enums.BalanceStatusEnum;
import org.springframework.lang.Nullable;

import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 负载均衡服务上下文
 */
@Data
public class BalanceServiceContext {

    private SyncRequest request;

    private boolean success = true;

    private WorkerMetaInfo selectedEngine;

    private boolean hitChatId = false;

    private WorkerMetaInfo cacheEngine;

    private BalanceStatusEnum statusEnum = BalanceStatusEnum.SUCCESS;

    private String errorMsg;

    private long startTimeInMs = System.currentTimeMillis();

    private long cacheRt;

    private long totalRt;

    private ConcurrentHashMap<String, WorkerStatus> workerStatuses;

    /**
     * 当前请求的模型下，GPU 类型和最佳吞吐并发的映射
     */
    @Nullable
    private Map<String/* GPU */, Integer> maxThroughputMap;

    /**
     * 选取的版本号
     */
    private String selectedVersion;

    /**
     * 获取 GPU 型号下在最大吞吐情况下的并发量，如果没有该数据，则返回 0
     *
     * @param gpu GPU 型号
     * @return 该 GPU 型号下，最大吞吐情况的并发令，如果没有该数据则返回 0
     */
    public int getMaxThroughputConcurrency(String gpu) {
        return Optional.ofNullable(maxThroughputMap)
                .map(map -> map.getOrDefault(gpu, 0))
                .orElse(0);
    }
}
