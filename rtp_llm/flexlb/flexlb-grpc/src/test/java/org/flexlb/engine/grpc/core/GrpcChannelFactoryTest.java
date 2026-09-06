package org.flexlb.engine.grpc.core;

import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import io.grpc.ConnectivityState;
import io.grpc.ManagedChannel;
import io.netty.bootstrap.Bootstrap;
import org.flexlb.config.ConfigService;
import org.flexlb.engine.grpc.config.ChannelConfiguration;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

@EnabledOnOs(OS.LINUX)
class GrpcChannelFactoryTest {

    @Test
    void createsChannelWithoutTcpUserTimeoutWarning() throws InterruptedException {
        Logger bootstrapLogger = (Logger) LoggerFactory.getLogger(Bootstrap.class);
        ListAppender<ILoggingEvent> appender = new ListAppender<>();
        appender.start();
        bootstrapLogger.addAppender(appender);

        try (AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(
                ConfigService.class, ChannelConfiguration.class, GrpcChannelFactory.class)) {
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
}
