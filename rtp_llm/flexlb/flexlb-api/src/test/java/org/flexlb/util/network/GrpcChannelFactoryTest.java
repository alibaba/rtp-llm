package org.flexlb.util.network;

import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import io.grpc.ConnectivityState;
import io.grpc.ManagedChannel;
import io.netty.bootstrap.Bootstrap;
import org.flexlb.api.config.ChannelConfiguration;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Checks configured channel creation and transport settings.
 */
class GrpcChannelFactoryTest {

    @Test
    void loadsChannelContextWithDefaultConfig() {
        try (AnnotationConfigApplicationContext context = createChannelContext()) {
            assertNotNull(context.getBean(GrpcChannelFactory.class));
        }
    }

    @Test
    void createsChannelWithoutTcpUserTimeoutWarning() throws Exception {
        Logger bootstrapLogger = (Logger) LoggerFactory.getLogger(Bootstrap.class);
        ListAppender<ILoggingEvent> appender = new ListAppender<>();
        appender.start();
        bootstrapLogger.addAppender(appender);

        io.grpc.Server server = io.grpc.netty.NettyServerBuilder.forPort(0).build().start();
        try (AnnotationConfigApplicationContext context = createChannelContext()) {
            GrpcChannelFactory factory = context.getBean(GrpcChannelFactory.class);
            ManagedChannel channel = factory.create(new GrpcTarget("127.0.0.1", server.getPort()));
            try {
                long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
                while (channel.getState(true) != ConnectivityState.READY) {
                    ConnectivityState state = channel.getState(true);
                    if (state == ConnectivityState.READY) break;
                    CountDownLatch stateChanged = new CountDownLatch(1);
                    channel.notifyWhenStateChanged(state, stateChanged::countDown);
                    assertTrue(
                            stateChanged.await(
                                    Math.max(1L, deadline - System.nanoTime()),
                                    TimeUnit.NANOSECONDS));
                }
            } finally {
                channel.shutdownNow();
                assertTrue(channel.awaitTermination(5, TimeUnit.SECONDS));
            }
        } finally {
            server.shutdownNow();
            assertTrue(server.awaitTermination(5, TimeUnit.SECONDS));
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

    private static AnnotationConfigApplicationContext createChannelContext() {
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext();
        context.getBeanFactory().registerSingleton("configService", configService);
        context.register(ChannelConfiguration.class, GrpcChannelFactory.class);
        context.refresh();
        return context;
    }
}
