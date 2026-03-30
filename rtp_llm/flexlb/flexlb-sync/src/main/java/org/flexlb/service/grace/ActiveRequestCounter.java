package org.flexlb.service.grace;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

@Component
public class ActiveRequestCounter {

    private static final Logger log = LoggerFactory.getLogger(ActiveRequestCounter.class);
    private final AtomicLong activeRequestCount = new AtomicLong(0);

    public RequestToken acquire() {
        activeRequestCount.incrementAndGet();
        return new RequestToken(this);
    }

    long release() {
        long previous = activeRequestCount.getAndUpdate(current -> current > 0 ? current - 1 : 0);
        if (previous == 0) {
            log.error("ActiveRequestCounter released below zero");
            return 0;
        }
        return previous - 1;
    }

    public long getCount() {
        return activeRequestCount.get();
    }

    public static final class RequestToken implements AutoCloseable {
        private final ActiveRequestCounter counter;
        private final AtomicBoolean closed = new AtomicBoolean(false);

        private RequestToken(ActiveRequestCounter counter) {
            this.counter = counter;
        }

        @Override
        public void close() {
            if (closed.compareAndSet(false, true)) {
                counter.release();
            }
        }
    }
}