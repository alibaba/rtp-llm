package org.flexlb.domain.batch;

import java.util.concurrent.DelayQueue;
import java.util.concurrent.Delayed;
import java.util.concurrent.TimeUnit;

/**
 * @author zjw
 * description:
 * date: 2025/3/13
 */
public class BoundedDelayQueue<T extends Delayed> extends DelayQueue<T> {

    private final int maxSize;

    public BoundedDelayQueue(int maxSize) {
        super();
        this.maxSize = maxSize;
    }

    @Override
    public boolean add(T t) {
        if (super.size() >= maxSize) {
            return false;
        }
        return super.add(t);
    }

    @Override
    public boolean offer(T t) {
        if (super.size() >= maxSize) {
            return false;
        }
        return super.offer(t);
    }

    @Override
    public void put(T t) {
        if (super.size() >= maxSize) {
            return;
        }
        super.put(t);
    }

    @Override
    public boolean offer(T t, long timeout, TimeUnit unit) {
        if (super.size() >= maxSize) {
            return false;
        }
        return super.offer(t, timeout, unit);
    }
}
