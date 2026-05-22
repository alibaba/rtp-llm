package org.flexlb.dispatcher;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class FePoolTest {

    @Test
    void roundRobinsAcrossAddresses() {
        FePool pool = new FePool(List.of("http://a:8088", "http://b:8088"));
        assertEquals("http://a:8088", pool.next());
        assertEquals("http://b:8088", pool.next());
        assertEquals("http://a:8088", pool.next());
    }

    @Test
    void reportsSize() {
        assertEquals(2, new FePool(List.of("http://a:8088", "http://b:8088")).size());
    }

    @Test
    void rejectsEmptyPool() {
        assertThrows(IllegalArgumentException.class, () -> new FePool(List.of()));
    }
}
