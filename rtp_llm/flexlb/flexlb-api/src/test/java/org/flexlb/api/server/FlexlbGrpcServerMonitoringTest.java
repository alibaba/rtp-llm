package org.flexlb.api.server;

import io.grpc.Server;
import io.micrometer.core.instrument.Meter;
import io.micrometer.core.instrument.config.MeterFilter;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import io.netty.channel.nio.NioEventLoopGroup;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.util.metrics.MetricConstant;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.core.env.Environment;
import org.springframework.test.util.ReflectionTestUtils;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.CALLS_REAL_METHODS;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Verifies executor metrics reach both the configured monitoring backend and the meter registry.
 */
class FlexlbGrpcServerMonitoringTest {

    @org.junit.jupiter.api.Test
    void businessBackendAndRegistryKeepAllOriginalExecutorSeries() {
        var backend = mock(org.flexlb.util.metrics.FlexMonitor.class);
        var registry = new SimpleMeterRegistry();
        var stats = mock(org.flexlb.util.metrics.ExecutorStats.class);
        var completed = new java.util.concurrent.atomic.AtomicLong(7);
        when(stats.activeThreads()).thenReturn(2);
        when(stats.queuedTasks()).thenReturn(3);
        when(stats.poolSize()).thenReturn(4);
        when(stats.completedTasks()).thenAnswer(call -> completed.get());
        try (var context =
                new org.springframework.context.annotation.AnnotationConfigApplicationContext()) {
            context.register(org.flexlb.config.monitoring.MonitorClientConfig.class);
            context.registerBean(
                    "businessMonitor", org.flexlb.util.metrics.FlexMonitor.class, () -> backend);
            context.registerBean(
                    "meterRegistry",
                    io.micrometer.core.instrument.MeterRegistry.class,
                    () -> registry);
            context.refresh();
            var monitor = context.getBean(org.flexlb.util.metrics.FlexMonitor.class);
            new org.flexlb.util.metrics.GrpcReporter(monitor)
                    .registerServerExecutor(stats, 8, completed::get);
            for (String name :
                    new String[] {
                        MetricConstant.GRPC_SERVER_EXECUTOR_ACTIVE_THREADS,
                        MetricConstant.GRPC_SERVER_EXECUTOR_QUEUE_SIZE,
                        MetricConstant.GRPC_SERVER_EXECUTOR_POOL_SIZE,
                        MetricConstant.GRPC_SERVER_EXECUTOR_MAX_POOL_SIZE,
                    }) {
                assertNotNull(registry.get("flexlb." + name).gauge());
            }
            for (long count : new long[] {7, 19}) {
                completed.set(count);
                for (String name :
                        new String[] {
                            MetricConstant.GRPC_SERVER_EXECUTOR_COMPLETED_TASKS,
                            MetricConstant.GRPC_SERVER_EXECUTOR_CALLER_RUNS
                        }) {
                    org.junit.jupiter.api.Assertions.assertEquals(
                            count, registry.get("flexlb." + name).functionCounter().count());
                }
            }
        } finally {
            registry.close();
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void brokenExecutorMetricCannotPreventServerStartup(boolean error) throws Exception {
        var config = new FlexlbConfig();
        config.getGrpcServer().setExecutorCoreSize(1);
        config.getGrpcServer().setExecutorMaxSize(1);
        config.getGrpcServer().setShutdownQuietPeriodMs(0);
        var configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        var environment = mock(Environment.class);
        when(environment.getProperty("server.port")).thenReturn("-2");
        var workerGroup = new NioEventLoopGroup(1);
        var registry = new SimpleMeterRegistry();
        registry.config()
                .meterFilter(
                        new MeterFilter() {
                            @Override
                            public Meter.Id map(Meter.Id id) {
                                if (id.getName()
                                        .equals(
                                                "flexlb."
                                                        + MetricConstant
                                                                .GRPC_SERVER_EXECUTOR_ACTIVE_THREADS)) {
                                    if (error) {
                                        throw new AssertionError("monitor failed");
                                    }
                                    throw new IllegalStateException("monitor failed");
                                }
                                return id;
                            }
                        });
        var service = mock(FlexlbServiceImpl.class, CALLS_REAL_METHODS);
        var server =
                new FlexlbGrpcServer(
                        service,
                        configService,
                        environment,
                        workerGroup,
                        new org.flexlb.util.metrics.GrpcReporter(
                                new org.flexlb.util.metrics.MicrometerFlexMonitor(registry)),
                        new GrpcServerTimingInterceptor(),
                        new GrpcQosHeaderInterceptor());
        try {
            assertDoesNotThrow(server::start);
            var runningServer = (Server) ReflectionTestUtils.getField(server, "server");
            assertNotNull(runningServer);
            assertTrue(runningServer.getPort() > 0);
            assertNotNull(
                    registry.find("flexlb." + MetricConstant.GRPC_SERVER_EXECUTOR_QUEUE_SIZE)
                            .gauge());
            assertNotNull(
                    registry.find("flexlb." + MetricConstant.GRPC_SERVER_EXECUTOR_COMPLETED_TASKS)
                            .functionCounter());
        } finally {
            server.shutdown();
            workerGroup.shutdownGracefully(0, 1, java.util.concurrent.TimeUnit.SECONDS).sync();
            registry.close();
        }
    }
}
