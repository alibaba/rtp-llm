package org.flexlb.service.grace;

import org.springframework.stereotype.Component;

import java.util.concurrent.atomic.LongAdder;

@Component
public class ActiveRequestCounter {

    private final LongAdder activeRequestCount = new LongAdder();

    public void increment() {
        activeRequestCount.increment();
    }

    public void decrement() {
        activeRequestCount.decrement();
    }

    public long getCount() {
        return activeRequestCount.sum();
    }
}