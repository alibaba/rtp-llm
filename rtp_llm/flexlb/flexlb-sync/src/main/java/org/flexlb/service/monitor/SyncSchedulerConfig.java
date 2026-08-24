package org.flexlb.service.monitor;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.resource.DynamicWorkerManager;
import org.flexlb.balance.scheduler.BatchScheduler;
import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.DirectScheduler;
import org.flexlb.balance.scheduler.InflightStore;
import org.flexlb.balance.scheduler.QueueScheduler;
import org.flexlb.config.ConfigService;
import org.flexlb.constant.MetricConstant;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.sync.shadow.StateShadowBridge;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.EnableScheduling;

import java.util.concurrent.ScheduledThreadPoolExecutor;

@Slf4j
@Configuration
@EnableScheduling
public class SyncSchedulerConfig {

    /**
     * Custom thread pool for Spring Boot scheduled tasks {@link org.springframework.scheduling.annotation.Scheduled}
     *
     * <p><b>NOTE:</b> Name must be {@code taskScheduler}</p>
     */
    @Bean(name = "taskScheduler")
    public ScheduledThreadPoolExecutor taskScheduler() {
        return new ScheduledThreadPoolExecutor(4, new NamedThreadFactory("task-scheduler"));
    }

    @Bean(name = "taskMetricScheduler")
    public ScheduledThreadPoolExecutor taskMetricScheduler() {
        return new ScheduledThreadPoolExecutor(1, new NamedThreadFactory("task-metric-scheduler"));
    }

    // ==================== Scheduler beans ====================

    /**
     * G1 影子门面单例 Bean：开关（flexlbStateV2ShadowEnabled / env
     * FLEXLB_STATE_V2_SHADOW_ENABLED）启动时定、不热切——关时返回
     * {@link StateShadowBridge#DISABLED} no-op 单例，注入各挂点的影子调用
     * 全部字节码级短路（零行为变化）；开时构建 StateLedger/Translator/
     * DiffCollector 并注册影子指标、打印启用回显。
     */
    @Bean
    public StateShadowBridge stateShadowBridge(ConfigService configService, FlexMonitor flexMonitor) {
        return StateShadowBridge.create(configService.loadBalanceConfig(), flexMonitor);
    }

    /**
     * BATCH-mode scheduler bean. Assembled here so that {@code RouteService}
     * can inject it directly instead of constructing it internally with
     * raw dependencies — the QUEUE-specific dependencies
     * ({@link RoutingQueueReporter}, {@link DynamicWorkerManager}) stay
     * inside this bean and never leak into {@code RouteService}.
     */
    @Bean
    public BatchScheduler batchScheduler(ConfigService configService,
                                        DefaultRouter router,
                                        EndpointRegistry endpointRegistry,
                                        BatchSchedulerReporter reporter,
                                        InflightStore globalInflightStore,
                                        FlexMonitor flexMonitor,
                                        StateShadowBridge shadowBridge) {
        FlexlbMetricHelper batchHelper = new FlexlbMetricHelper(flexMonitor, MetricConstant.PATH_BATCH);
        batchHelper.register();
        return new BatchScheduler(configService, router, endpointRegistry, reporter,
                globalInflightStore, batchHelper, shadowBridge);
    }

    /**
     * QUEUE-mode scheduler bean. {@link RoutingQueueReporter} and
     * {@link DynamicWorkerManager} are injected here, keeping them out of
     * {@code RouteService}'s constructor.
     */
    @Bean
    public QueueScheduler queueScheduler(DefaultRouter router,
                                         ConfigService configService,
                                         RoutingQueueReporter routingQueueReporter,
                                         DynamicWorkerManager dynamicWorkerManager,
                                         InflightStore globalInflightStore,
                                         FlexMonitor flexMonitor) {
        FlexlbMetricHelper queueHelper = new FlexlbMetricHelper(flexMonitor, MetricConstant.PATH_QUEUE);
        queueHelper.register();
        return new QueueScheduler(router, configService, routingQueueReporter,
                dynamicWorkerManager, globalInflightStore, queueHelper);
    }

    /**
     * DIRECT-mode scheduler bean.
     */
    @Bean
    public DirectScheduler directScheduler(DefaultRouter router,
                                           InflightStore globalInflightStore,
                                           FlexMonitor flexMonitor) {
        FlexlbMetricHelper directHelper = new FlexlbMetricHelper(flexMonitor, MetricConstant.PATH_DIRECT);
        directHelper.register();
        return new DirectScheduler(router, globalInflightStore, directHelper);
    }
}
