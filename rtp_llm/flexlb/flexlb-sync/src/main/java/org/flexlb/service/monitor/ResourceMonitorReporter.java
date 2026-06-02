package org.flexlb.service.monitor;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.resource.DynamicWorkerManager;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;

import static org.flexlb.constant.MetricConstant.WORKER_PERMIT_CAPACITY;

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
    private final FlexMetricTags tags = FlexMetricTags.of();

    @Autowired
    public ResourceMonitorReporter(FlexMonitor monitor, DynamicWorkerManager dynamicWorkerManager) {
        this.monitor = monitor;
        this.dynamicWorkerManager = dynamicWorkerManager;
    }

    @PostConstruct
    public void init() {
        monitor.register(WORKER_PERMIT_CAPACITY, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        log.info("ResourceMonitorReporter initialized and registered with KMonitor");
    }

    @Scheduled(fixedRate = 1000)
    private void reportWorkerPermitCapacity() {
        try {
            int capacity = dynamicWorkerManager.getTotalPermits();
            monitor.report(WORKER_PERMIT_CAPACITY, tags, capacity);
        } catch (Exception e) {
            log.error("Failed to report worker permit capacity", e);
        }
    }
}