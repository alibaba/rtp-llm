package org.flexlb.service.monitor;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.resource.ResourceMonitor;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;

import static org.flexlb.constant.MetricConstant.RESOURCE_AVAILABLE_STATUS;

/**
 * 资源监控指标器
 *
 * @author saichen.sm
 * @since 2025/12/23
 */
@Slf4j
@Component
public class ResourceMonitorReporter {

    private final FlexMonitor monitor;
    private final ResourceMonitor resourceMonitor;
    private final FlexMetricTags tags = FlexMetricTags.of();

    @Autowired
    public ResourceMonitorReporter(FlexMonitor monitor, ResourceMonitor resourceMonitor) {
        this.monitor = monitor;
        this.resourceMonitor = resourceMonitor;
    }

    @PostConstruct
    public void init() {
        monitor.register(RESOURCE_AVAILABLE_STATUS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        log.info("ResourceMonitorReporter initialized and registered with KMonitor");
    }

    @Scheduled(fixedRate = 1000)
    private void reportResourceStatus() {
        try {
            double status = resourceMonitor.hasAvailableResource() ? 1.0 : 0.0;
            monitor.report(RESOURCE_AVAILABLE_STATUS, tags, status);
        } catch (Exception e) {
            log.error("Failed to report resource status", e);
        }
    }
}