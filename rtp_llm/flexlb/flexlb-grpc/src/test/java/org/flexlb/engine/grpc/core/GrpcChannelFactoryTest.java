package org.flexlb.engine.grpc.core;

import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import io.grpc.ConnectivityState;
import io.grpc.ManagedChannel;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import io.netty.bootstrap.Bootstrap;
import io.netty.channel.EventLoopGroup;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.engine.grpc.config.ChannelConfiguration;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.core.env.MapPropertySource;

import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.stream.StreamSupport;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Checks configured channel creation and transport settings.
 */
class GrpcChannelFactoryTest {

    @Test
    void loadsChannelContextWithDefaultConfig() {
        try (AnnotationConfigApplicationContext context = createChannelContext(Map.of())) {
            GrpcChannelFactory factory = context.getBean(GrpcChannelFactory.class);
            assertNotNull(factory);
            assertEngineGrpcSizing(context, 32, 10_000, 8);
            assertEquals(2_000L, factory.getKeepAliveTimeMillis());
            assertTrue(factory.isKeepAliveWithoutCalls());
        }
    }

    @Test
    void appliesEngineGrpcExecutorSizingToChannelBeans() {
        var config = new FlexlbConfig();
        var engineGrpc = config.getEngineGrpc();
        engineGrpc.setExecutorThreads(3);
        engineGrpc.setExecutorQueueCapacity(17);
        engineGrpc.setEventLoopThreads(2);
        try (var context = createChannelContext(Map.of(), config)) {
            assertEngineGrpcSizing(context, 3, 17, 2);
        }
    }

    @Test
    void loadsEngineKeepaliveOverrides() {
        try (AnnotationConfigApplicationContext context =
                createChannelContext(
                        Map.of(
                                "flexlb.engine-grpc.keep-alive-time-ms", "30000",
                                "flexlb.engine-grpc.keep-alive-without-calls", "false"))) {
            GrpcChannelFactory factory = context.getBean(GrpcChannelFactory.class);
            assertEquals(30_000L, factory.getKeepAliveTimeMillis());
            assertFalse(factory.isKeepAliveWithoutCalls());
        }
    }

    @Test
    void appliesConfiguredLimitOnlyToKvCacheStatusChannels() throws Exception {
        var response =
                EngineRpcService.CacheStatusPB.newBuilder()
                        .setMultimodalCache(
                                EngineRpcService.MultimodalCacheStatusPB.newBuilder()
                                        .addKeys("x".repeat(1_024)))
                        .build();
        var server =
                NettyServerBuilder.forPort(0)
                        .addService(
                                new RpcServiceGrpc.RpcServiceImplBase() {
                                    @Override
                                    public void getCacheStatus(
                                            EngineRpcService.CacheVersionPB request,
                                            StreamObserver<EngineRpcService.CacheStatusPB>
                                                    observer) {
                                        observer.onNext(response);
                                        observer.onCompleted();
                                    }
                                })
                        .build()
                        .start();
        var config = new FlexlbConfig();
        config.getEngineGrpc().setKvCacheMaxInboundMessageBytes(512);
        try (var context = createChannelContext(Map.of(), config)) {
            var factory = context.getBean(GrpcChannelFactory.class);
            var target = new GrpcTarget("127.0.0.1", server.getPort());
            var request = EngineRpcService.CacheVersionPB.getDefaultInstance();
            var regularChannel = factory.create(target);
            var cacheChannel = factory.createKvCacheStatus(target);
            try {
                var cacheStub =
                        RpcServiceGrpc.newBlockingStub(cacheChannel)
                                .withDeadlineAfter(5, TimeUnit.SECONDS);
                var error =
                        assertThrows(
                                StatusRuntimeException.class,
                                () -> cacheStub.getCacheStatus(request));
                assertEquals(Status.Code.RESOURCE_EXHAUSTED, error.getStatus().getCode());
                assertEquals(
                        response,
                        RpcServiceGrpc.newBlockingStub(regularChannel)
                                .withDeadlineAfter(5, TimeUnit.SECONDS)
                                .getCacheStatus(request));
            } finally {
                regularChannel.shutdownNow();
                cacheChannel.shutdownNow();
            }
        } finally {
            server.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    @Test
    void channelExecutorsStillRunLateCallbacksAfterShutdown() {
        try (var context = createChannelContext(Map.of())) {
            for (String beanName :
                    new String[] {"managedChannelThreadPoolExecutor", "forwarderChannelExecutor"}) {
                var executor = context.getBean(beanName, ThreadPoolExecutor.class);
                var ran = new CountDownLatch(1);
                executor.shutdown();

                executor.execute(ran::countDown);

                assertEquals(0, ran.getCount(), beanName);
            }
        }
    }

    @Test
    @EnabledOnOs(OS.LINUX)
    void createsChannelWithoutTcpUserTimeoutWarning() throws InterruptedException {
        Logger bootstrapLogger = (Logger) LoggerFactory.getLogger(Bootstrap.class);
        ListAppender<ILoggingEvent> appender = new ListAppender<>();
        appender.start();
        bootstrapLogger.addAppender(appender);

        try (AnnotationConfigApplicationContext context = createChannelContext(Map.of())) {
            GrpcChannelFactory factory = context.getBean(GrpcChannelFactory.class);
            ManagedChannel channel = factory.create(new GrpcTarget("127.0.0.1", 1));
            try {
                CountDownLatch stateChanged = new CountDownLatch(1);
                channel.notifyWhenStateChanged(ConnectivityState.IDLE, stateChanged::countDown);
                channel.getState(true);
                assertTrue(stateChanged.await(2, TimeUnit.SECONDS));
            } finally {
                channel.shutdownNow();
            }
        } finally {
            bootstrapLogger.detachAppender(appender);
            appender.stop();
        }

        assertFalse(
                appender.list.stream()
                        .map(ILoggingEvent::getFormattedMessage)
                        .anyMatch(
                                message ->
                                        message.contains("Unknown channel option")
                                                && message.contains("TCP_USER_TIMEOUT")));
    }

    private static void assertEngineGrpcSizing(
            AnnotationConfigApplicationContext context, int threads, int capacity, int ioThreads) {
        var executor =
                context.getBean("managedChannelThreadPoolExecutor", ThreadPoolExecutor.class);
        assertEquals(threads, executor.getCorePoolSize());
        assertEquals(threads, executor.getMaximumPoolSize());
        assertEquals(capacity, executor.getQueue().remainingCapacity());
        var eventLoops = context.getBean("managedChannelEventLoopGroup", EventLoopGroup.class);
        assertEquals(ioThreads, StreamSupport.stream(eventLoops.spliterator(), false).count());
    }

    private static AnnotationConfigApplicationContext createChannelContext(
            Map<String, Object> properties) {
        return createChannelContext(properties, new FlexlbConfig());
    }

    private static AnnotationConfigApplicationContext createChannelContext(
            Map<String, Object> properties, FlexlbConfig config) {
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext();
        context.getEnvironment()
                .getPropertySources()
                .addFirst(new MapPropertySource("test", properties));
        context.getBeanFactory().registerSingleton("configService", configService);
        context.register(ChannelConfiguration.class, GrpcChannelFactory.class);
        context.refresh();
        return context;
    }
}
