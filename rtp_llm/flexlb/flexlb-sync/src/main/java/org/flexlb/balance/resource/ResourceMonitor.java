package org.flexlb.balance.resource;

import lombok.Setter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.WhaleMasterConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.flexlb.util.LoggingUtils;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 资源监控器: 监控资源可用性, 负责检测资源状态变化并通知监听器
 *
 * @author saichen.sm
 * @since 2025/12/23
 */
@Component
public class ResourceMonitor {

    private final ResourceMeasureFactory resourceMeasureFactory;
    private final ConfigService configService;

    // 资源检查线程 (周期性检查资源可用性)
    private ScheduledExecutorService resourceChecker;

    // 全局资源可用性标志
    private final AtomicBoolean hasAvailableResourceFlag = new AtomicBoolean(true);

    // 资源可用事件监听器
    @Setter
    private ResourceAvailableListener resourceAvailableListener;

    public ResourceMonitor(ResourceMeasureFactory resourceMeasureFactory,
                           ConfigService configService) {
        this.resourceMeasureFactory = resourceMeasureFactory;
        this.configService = configService;
    }

    @PostConstruct
    public void startScheduler() {
        WhaleMasterConfig config = configService.loadBalanceConfig();

        // 启动资源检查线程 (周期性检查资源可用性)
        this.resourceChecker = Executors.newScheduledThreadPool(1, r -> {
            Thread t = new Thread(r, "routing-resource-checker");
            t.setDaemon(true);
            return t;
        });
        resourceChecker.scheduleWithFixedDelay(
                this::checkAllResourceAvailable,
                config.getResourceCheckIntervalMs(),
                config.getResourceCheckIntervalMs(),
                TimeUnit.MILLISECONDS
        );
        LoggingUtils.info("QueueManager Resource checker started, interval: {}ms",
                config.getResourceCheckIntervalMs());
    }

    /**
     * 检查所有引擎机器的资源可用性
     * 当有资源可用时,通知调度器
     * 资源检查(周期调度):
     *     └─ 如果有可用资源:
     *        ├─ hasAvailableResourceFlag.set(true)
     *        └─ notifyResourceAvailable()          ← 发送资源可用信号
     *     └─ 如果无可用资源:
     *        └─ hasAvailableResourceFlag.set(false)
     */
    public void checkAllResourceAvailable() {
        try {
            boolean anyResourceAvailable = false;

            // 检查所有模型的所有 Role 是否有可用资源
            for (Map.Entry<String, ModelWorkerStatus> entry : EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP.entrySet()) {
                String modelName = entry.getKey();
                ModelWorkerStatus workerStatus = entry.getValue();

                List<RoleType> roleTypeList = workerStatus.getRoleTypeList();
                if (hasAvailableResources(modelName, roleTypeList)) {
                    anyResourceAvailable = true;
                    break;
                }
            }

            // 只要资源可用就发送信号
            if (anyResourceAvailable) {
                hasAvailableResourceFlag.set(true);
                notifyResourceAvailable();
            } else {
                // 资源不可用时,更新标志位
                hasAvailableResourceFlag.set(false);
            }

        } catch (Exception e) {
            LoggingUtils.error("Resource checker encountered error", e);
        }
    }

    /**
     * 检查单个引擎机器的资源可用性
     * 当有资源可用时,通知调度器
     *
     * @param workerStatus worker状态
     */
    public void checkSingleResourceAvailable(WorkerStatus workerStatus) {
        try {
            // 当全局资源可用时, 无需检查
            if (hasAvailableResourceFlag.get()) {
                return;
            }
            RoleType roleType = RoleType.getBy(workerStatus.getRole());
            ResourceMeasure resourceMeasure = resourceMeasureFactory.getMeasure(roleType.getResourceMeasureIndicator());

            boolean workerHasResource = resourceMeasure.isResourceAvailable(workerStatus);

            // 检测到该worker资源满足
            if (workerHasResource) {
                hasAvailableResourceFlag.set(true);
                notifyResourceAvailable();
            }
        } catch (Exception e) {
            LoggingUtils.error("onWorkerStatusChange error for worker {}:{}, role:{}",
                    workerStatus.getIp(), workerStatus.getPort(), workerStatus.getRole(), e);
        }
    }

    /**
     * 检查指定模型和角色列表是否有可用资源
     *
     * @param modelName    模型名称
     * @param roleTypeList 角色类型列表
     * @return true表示所有角色都有可用Worker
     */
    public boolean hasAvailableResources(String modelName, List<RoleType> roleTypeList) {
        for (RoleType roleType : roleTypeList) {
            ResourceMeasureIndicatorEnum indicator = roleType.getResourceMeasureIndicator();
            ResourceMeasure resourceMeasure = resourceMeasureFactory.getMeasure(indicator);
            if (!resourceMeasure.hasResourceAvailableWorker(modelName, null)) {
                return false;
            }
        }
        return true;
    }

    /**
     * 检查全局资源是否可用
     * 供 RequestScheduler 实时检查使用
     *
     * @return true表示当前有可用资源
     */
    public boolean hasAvailableResource() {
        return hasAvailableResourceFlag.get();
    }

    /**
     * 通知资源监听器
     */
    private void notifyResourceAvailable() {
        if (resourceAvailableListener == null) {
            return;
        }
        resourceAvailableListener.onResourceAvailable();
    }

    @PreDestroy
    public void shutdown() {
        // 关闭资源检查线程
        if (resourceChecker != null && !resourceChecker.isShutdown()) {
            resourceChecker.shutdown();
            try {
                if (!resourceChecker.awaitTermination(5, TimeUnit.SECONDS)) {
                    resourceChecker.shutdownNow();
                }
                LoggingUtils.info("QueueManager resource checker stopped");
            } catch (InterruptedException e) {
                resourceChecker.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }
}
