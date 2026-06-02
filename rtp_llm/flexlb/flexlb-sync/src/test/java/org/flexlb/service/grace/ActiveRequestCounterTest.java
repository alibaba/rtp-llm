package org.flexlb.service.grace;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ActiveRequestCounterTest {

    @Test
    void requestTokenCloseShouldBeIdempotent() {
        ActiveRequestCounter counter = new ActiveRequestCounter();

        ActiveRequestCounter.RequestToken token = counter.acquire();
        assertEquals(1, counter.getCount());

        token.close();
        assertEquals(0, counter.getCount());

        token.close();
        assertEquals(0, counter.getCount());
    }

    @Test
    void releaseShouldNotUnderflowCounter() {
        ActiveRequestCounter counter = new ActiveRequestCounter();

        assertEquals(0, counter.release());
        assertEquals(0, counter.getCount());
    }
}
