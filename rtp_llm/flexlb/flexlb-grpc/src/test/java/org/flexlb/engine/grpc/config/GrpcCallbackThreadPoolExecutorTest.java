package org.flexlb.engine.grpc.config;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GrpcCallbackThreadPoolExecutorTest {

    @Test
    void logsAndCountsTaskRejectionWhenAllCallbackThreadsAreBusy() throws InterruptedException {
        GrpcCallbackThreadPoolExecutor executor = new GrpcCallbackThreadPoolExecutor(
                1, 1, 1, TimeUnit.MINUTES, new ArrayBlockingQueue<>(1),
                new NamedThreadFactory("grpc-callback-test"));
        CountDownLatch taskStarted = new CountDownLatch(1);
        CountDownLatch releaseTask = new CountDownLatch(1);
        Logger logger = (Logger) LoggerFactory.getLogger(GrpcCallbackThreadPoolExecutor.class);
        ListAppender<ILoggingEvent> appender = new ListAppender<>();
        appender.start();
        logger.addAppender(appender);
        try {
            executor.execute(() -> {
                taskStarted.countDown();
                try {
                    releaseTask.await();
                } catch (InterruptedException interruptedException) {
                    Thread.currentThread().interrupt();
                }
            });
            assertTrue(taskStarted.await(1, TimeUnit.SECONDS));
            executor.execute(() -> { });

            assertThrows(RejectedExecutionException.class, () -> executor.execute(() -> { }));
            assertThrows(RejectedExecutionException.class, () -> executor.execute(() -> { }));

            assertEquals(2, executor.getRejectedTaskCount());
            assertEquals(1, appender.list.stream()
                    .filter(event -> event.getLevel() == Level.ERROR
                            && event.getFormattedMessage().contains("gRPC callback executor rejected task"))
                    .count());
        } finally {
            releaseTask.countDown();
            executor.shutdownNow();
            logger.detachAppender(appender);
        }
    }
}
