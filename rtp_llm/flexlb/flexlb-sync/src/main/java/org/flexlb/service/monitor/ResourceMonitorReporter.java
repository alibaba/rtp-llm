package org.flexlb.service.monitor;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.resource.DynamicWorkerManager;
import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.Map;

import static org.flexlb.constant.MetricConstant.WORKER_PERMIT_CAPACITY;
import static org.flexlb.constant.MetricConstant.WORKER_RESOURCE_WATER_LEVEL;

/**
 * Worker permit capacity monitoring reporter
 *
 * @author saichen.sm
 * @since 2025/12/23
 */
@Slf4j
@Component
public class ResourceMonitorReporter {

    private final FlexMonitor monitor;
    private final DynamicWorkerManager dynamicWorkerManager;
    private final ConfigService configService;
    private final ResourceMeasureFactory resourceMeasureFactory;
    private final FlexMetricTags tags = FlexMetricTags.of();

    @Autowired
    public ResourceMonitorReporter(FlexMonitor monitor,
                                   DynamicWorkerManager dynamicWorkerManager,
                                   ConfigService configService,
                                   ResourceMeasureFactory resourceMeasureFactory) {
        this.monitor = monitor;
        this.dynamicWorkerManager = dynamicWorkerManager;
        this.configService = configService;
        this.resourceMeasureFactory = resourceMeasureFactory;
    }

    @PostConstruct
    public void init() {
        monitor.register(WORKER_PERMIT_CAPACITY, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(WORKER_RESOURCE_WATER_LEVEL, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        log.info("ResourceMonitorReporter initialized and registered with KMonitor");
    }

    @Scheduled(fixedRate = 1000)
    void reportResourceMetrics() {
        try {
            reportWorkerPermitCapacity();
            reportWorkerWaterLevels();
        } catch (Exception e) {
            log.error("Failed to report FlexLB resource metrics", e);
        }
    }

    private void reportWorkerPermitCapacity() {
        int capacity = dynamicWorkerManager.getTotalPermits();
        monitor.report(WORKER_PERMIT_CAPACITY, tags, capacity);
    }

    void reportWorkerWaterLevels() {
        ModelWorkerStatus modelWorkerStatus = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS;

        FlexlbConfig config = configService.loadBalanceConfig();
        for (RoleType roleType : modelWorkerStatus.getRoleTypeList()) {
            ResourceMeasureIndicatorEnum indicator = config.getResourceMeasureIndicator(roleType);
            ResourceMeasure measure = resourceMeasureFactory.getMeasure(indicator);
            if (measure == null) {
                continue;
            }

            Map<String, WorkerStatus> workerStatuses = modelWorkerStatus.getRoleStatusMap(roleType);
            for (WorkerStatus workerStatus : workerStatuses.values()) {
                if (workerStatus == null || workerStatus.getIp() == null || workerStatus.getIp().isBlank()) {
                    continue;
                }
                FlexMetricTags workerTags = FlexMetricTags.of(
                        "engineIp", workerStatus.getIpIndex(),
                        "role", roleType.getCode());
                monitor.report(WORKER_RESOURCE_WATER_LEVEL,
                        workerTags,
                        measure.calculateWorkerWaterLevel(workerStatus));
            }
        }
    }
}
