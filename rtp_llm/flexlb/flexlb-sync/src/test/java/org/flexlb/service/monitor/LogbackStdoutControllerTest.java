package org.flexlb.service.monitor;

import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.Appender;
import org.junit.jupiter.api.Test;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

class LogbackStdoutControllerTest {

    @Test
    void attachesAndDetachesConsoleAppender() {
        Logger rootLogger = mock(Logger.class);
        Logger pvLogger = mock(Logger.class);
        @SuppressWarnings("unchecked")
        Appender<ILoggingEvent> consoleAppender = mock(Appender.class);
        LogbackStdoutController controller =
                new LogbackStdoutController(rootLogger, pvLogger, consoleAppender);

        controller.setEnabled(false);
        controller.setEnabled(true);

        verify(rootLogger).detachAppender(consoleAppender);
        verify(pvLogger).detachAppender(consoleAppender);
        verify(rootLogger).addAppender(consoleAppender);
        verify(pvLogger).addAppender(consoleAppender);
    }
}
