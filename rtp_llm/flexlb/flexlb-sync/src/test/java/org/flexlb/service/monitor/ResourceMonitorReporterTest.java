package org.flexlb.service.monitor;

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
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.ArrayList;
import java.util.List;

import static org.flexlb.constant.MetricConstant.WORKER_RESOURCE_WATER_LEVEL;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class ResourceMonitorReporterTest {

    @Mock
    private DynamicWorkerManager dynamicWorkerManager;
    @Mock
    private ConfigService configService;
    @Mock
    private ResourceMeasureFactory resourceMeasureFactory;
    @Mock
    private ResourceMeasure prefillMeasure;
    @Mock
    private ResourceMeasure decodeMeasure;

    @AfterEach
    void clearWorkerStatuses() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().clear();
    }

    @Test
    void reportsWaterLevelForEachEngineAndRole() {
        CapturingFlexMonitor monitor = new CapturingFlexMonitor();
        FlexlbConfig config = new FlexlbConfig();
        when(configService.loadBalanceConfig()).thenReturn(config);
        when(resourceMeasureFactory.getMeasure(ResourceMeasureIndicatorEnum.WAIT_TIME))
                .thenReturn(prefillMeasure);
        when(resourceMeasureFactory.getMeasure(ResourceMeasureIndicatorEnum.REMAINING_KV_CACHE))
                .thenReturn(decodeMeasure);

        WorkerStatus prefill = worker("10.0.0.1");
        WorkerStatus decode = worker("10.0.0.2");
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().put("10.0.0.1:8080", prefill);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().put("10.0.0.2:8080", decode);
        when(prefillMeasure.calculateWorkerWaterLevel(prefill)).thenReturn(25.0);
        when(decodeMeasure.calculateWorkerWaterLevel(decode)).thenReturn(60.0);

        ResourceMonitorReporter reporter = new ResourceMonitorReporter(
                monitor, dynamicWorkerManager, configService, resourceMeasureFactory);
        reporter.init();
        reporter.reportWorkerWaterLevels();

        assertEquals(FlexMetricType.GAUGE, monitor.registeredTypes.get(WORKER_RESOURCE_WATER_LEVEL));
        assertEquals(FlexPriorityType.PRECISE, monitor.registeredPriorities.get(WORKER_RESOURCE_WATER_LEVEL));
        assertEquals(List.of(
                new Report(WORKER_RESOURCE_WATER_LEVEL,
                        FlexMetricTags.of("engineIp", "10.0.0.2", "role", RoleType.DECODE.getCode()),
                        60.0),
                new Report(WORKER_RESOURCE_WATER_LEVEL,
                        FlexMetricTags.of("engineIp", "10.0.0.1", "role", RoleType.PREFILL.getCode()),
                        25.0)), monitor.reports);
    }

    private static WorkerStatus worker(String ip) {
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp(ip);
        return workerStatus;
    }

    private record Report(String metricName, FlexMetricTags tags, double value) {
    }

    private static final class CapturingFlexMonitor implements FlexMonitor {
        private final java.util.Map<String, FlexMetricType> registeredTypes = new java.util.HashMap<>();
        private final java.util.Map<String, FlexPriorityType> registeredPriorities = new java.util.HashMap<>();
        private final List<Report> reports = new ArrayList<>();

        @Override
        public void register(String metricName, FlexMetricType metricType) {
            registeredTypes.put(metricName, metricType);
        }

        @Override
        public void register(String metricName, FlexMetricType metricType, FlexPriorityType priorityType) {
            registeredTypes.put(metricName, metricType);
            registeredPriorities.put(metricName, priorityType);
        }

        @Override
        public void register(String metricName, FlexMetricType metricType, int statisticsType) {
            registeredTypes.put(metricName, metricType);
        }

        @Override
        public void report(String metricName, double value) {
        }

        @Override
        public void report(String metricName, FlexMetricTags metricsTags, double value) {
            reports.add(new Report(metricName, metricsTags, value));
        }
    }
}
