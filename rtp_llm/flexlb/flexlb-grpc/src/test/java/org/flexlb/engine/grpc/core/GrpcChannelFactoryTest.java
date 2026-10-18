package org.flexlb.engine.grpc.core;

import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import io.grpc.ConnectivityState;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.netty.NettyServerBuilder;
import io.netty.bootstrap.Bootstrap;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.engine.grpc.config.ChannelConfiguration;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.BeanCreationException;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.core.NestedExceptionUtils;
import org.springframework.core.env.MapPropertySource;

import java.net.InetSocketAddress;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class GrpcChannelFactoryTest {

    @Test
    void loadsChannelContextWithDefaultConfig() {
        try (AnnotationConfigApplicationContext context = createChannelContext()) {
            assertNotNull(context.getBean(GrpcChannelFactory.class));
        }
    }

    @Test
    @EnabledOnOs(OS.LINUX)
    void createsChannelWithoutTcpUserTimeoutWarning() throws InterruptedException {
        Logger bootstrapLogger = (Logger) LoggerFactory.getLogger(Bootstrap.class);
        ListAppender<ILoggingEvent> appender = new ListAppender<>();
        appender.start();
        bootstrapLogger.addAppender(appender);

        try (AnnotationConfigApplicationContext context = createChannelContext()) {
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

        assertFalse(appender.list.stream()
                .map(ILoggingEvent::getFormattedMessage)
                .anyMatch(message -> message.contains("Unknown channel option")
                        && message.contains("TCP_USER_TIMEOUT")));
    }

    @Test
    void keepAliveDefaultsKeepTheOriginalChannelBehavior() {
        try (AnnotationConfigApplicationContext context = createChannelContext()) {
            GrpcChannelFactory factory = context.getBean(GrpcChannelFactory.class);
            assertEquals(2_000L, factory.getKeepAliveTimeMillis());
            assertTrue(factory.isKeepAliveWithoutCalls());
        }
    }

    @Test
    void keepAliveIsConfiguredFromPropertiesAndValidated() {
        try (AnnotationConfigApplicationContext context = createChannelContext(Map.of(
                GrpcChannelFactory.KEEP_ALIVE_TIME_PROPERTY, "30000",
                GrpcChannelFactory.KEEP_ALIVE_WITHOUT_CALLS_PROPERTY, "false"))) {
            GrpcChannelFactory factory = context.getBean(GrpcChannelFactory.class);
            assertEquals(30_000L, factory.getKeepAliveTimeMillis());
            assertFalse(factory.isKeepAliveWithoutCalls());
        }
        for (String invalid : new String[]{"0", "-1"}) {
            BeanCreationException error = assertThrows(BeanCreationException.class,
                    () -> createChannelContext(Map.of(GrpcChannelFactory.KEEP_ALIVE_TIME_PROPERTY, invalid)));
            assertTrue(NestedExceptionUtils.getMostSpecificCause(error).getMessage().contains("keepAliveTimeMillis"),
                    error.getMessage());
        }
    }

    /**
     * An engine whose gRPC server keeps the default keepalive rules (dashllm's worker-status server is
     * built that way) answers a ping on a connection with no running call with GOAWAY(too_many_pings)
     * on the third ping. A channel that pings while idle sends one every 10 seconds (grpc-java raises a
     * shorter interval to 10 seconds), so the server closes it after about 30 seconds. The default keeps
     * that behavior, which rtp-llm's tolerant server does not mind; turning the setting off avoids it.
     */
    @Test
    void keepAliveWithoutCallsSettingDecidesWhetherAnIdleChannelSurvivesDefaultServerRules() throws Exception {
        Server server = NettyServerBuilder.forAddress(new InetSocketAddress("127.0.0.1", 0))
                .build()
                .start();
        try (AnnotationConfigApplicationContext pingingContext = createChannelContext();
             AnnotationConfigApplicationContext quietContext = createChannelContext(
                     Map.of(GrpcChannelFactory.KEEP_ALIVE_WITHOUT_CALLS_PROPERTY, "false"))) {
            GrpcTarget target = new GrpcTarget("127.0.0.1", server.getPort());
            ManagedChannel pinging = pingingContext.getBean(GrpcChannelFactory.class).create(target);
            ManagedChannel quiet = quietContext.getBean(GrpcChannelFactory.class).create(target);
            try {
                waitUntilReady(pinging);
                waitUntilReady(quiet);
                boolean pingingLeftReady = stateChangesWithin(pinging, ConnectivityState.READY, 40_000);
                assertTrue(pingingLeftReady, "default channel should be closed by the server's too_many_pings");
                assertEquals(ConnectivityState.READY, quiet.getState(false),
                        "channel that does not ping while idle was closed by the server");
            } finally {
                pinging.shutdownNow();
                quiet.shutdownNow();
            }
        } finally {
            server.shutdownNow();
        }
    }

    private static void waitUntilReady(ManagedChannel channel) throws InterruptedException {
        ConnectivityState state = channel.getState(true);
        while (state != ConnectivityState.READY) {
            assertTrue(stateChangesWithin(channel, state, 5_000), "channel did not become READY, state=" + state);
            state = channel.getState(false);
        }
    }

    private static boolean stateChangesWithin(ManagedChannel channel, ConnectivityState currentState, long timeoutMillis)
            throws InterruptedException {
        CountDownLatch stateChanged = new CountDownLatch(1);
        channel.notifyWhenStateChanged(currentState, stateChanged::countDown);
        return stateChanged.await(timeoutMillis, TimeUnit.MILLISECONDS);
    }

    private static AnnotationConfigApplicationContext createChannelContext() {
        return createChannelContext(Map.of());
    }

    private static AnnotationConfigApplicationContext createChannelContext(Map<String, Object> properties) {
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext();
        context.getEnvironment().getPropertySources().addFirst(new MapPropertySource("test", properties));
        context.getBeanFactory().registerSingleton("configService", configService);
        context.register(ChannelConfiguration.class, GrpcChannelFactory.class);
        context.refresh();
        return context;
    }
}
