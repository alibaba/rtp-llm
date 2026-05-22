package org.flexlb.dispatcher;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class FePool {

    private final List<String> addresses;
    private final AtomicInteger cursor = new AtomicInteger(0);

    public FePool(List<String> addresses) {
        if (addresses == null || addresses.isEmpty()) {
            throw new IllegalArgumentException("FE pool must not be empty");
        }
        this.addresses = List.copyOf(addresses);
    }

    public String next() {
        return addresses.get(Math.floorMod(cursor.getAndIncrement(), addresses.size()));
    }

    public int size() {
        return addresses.size();
    }
}
