package org.flexlb.httpserver;

import io.grpc.Server;
import io.micrometer.core.instrument.Meter;
import io.micrometer.core.instrument.config.MeterFilter;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import io.netty.channel.nio.NioEventLoopGroup;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.constant.MetricConstant;
import org.flexlb.interceptor.GrpcQosHeaderInterceptor;
import org.flexlb.interceptor.GrpcServerTimingInterceptor;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.core.env.Environment;
import org.springframework.test.util.ReflectionTestUtils;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class FlexlbGrpcServerMonitoringTest {

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
        registry.config().meterFilter(new MeterFilter() {
            @Override
            public Meter.Id map(Meter.Id id) {
                if (id.getName().equals("flexlb." + MetricConstant.GRPC_SERVER_EXECUTOR_ACTIVE_THREADS)) {
                    if (error) {
                        throw new AssertionError("monitor failed");
                    }
                    throw new IllegalStateException("monitor failed");
                }
                return id;
            }
        });
        var service = mock(FlexlbServiceImpl.class, CALLS_REAL_METHODS);
        var server = new FlexlbGrpcServer(service, configService, environment, workerGroup,
                registry, new GrpcServerTimingInterceptor(), new GrpcQosHeaderInterceptor());
        try {
            assertDoesNotThrow(server::start);
            var runningServer = (Server) ReflectionTestUtils.getField(server, "server");
            assertNotNull(runningServer);
            assertTrue(runningServer.getPort() > 0);
            assertNotNull(registry.find("flexlb." + MetricConstant.GRPC_SERVER_EXECUTOR_QUEUE_SIZE).gauge());
            assertNotNull(registry.find("flexlb." + MetricConstant.GRPC_SERVER_EXECUTOR_COMPLETED_TASKS).functionCounter());
        } finally {
            server.shutdown();
            workerGroup.shutdownGracefully(0, 1, java.util.concurrent.TimeUnit.SECONDS).sync();
            registry.close();
        }
    }
}
