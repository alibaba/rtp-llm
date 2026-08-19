package org.flexlb.sync.status;

import com.google.common.util.concurrent.Striped;
import org.springframework.stereotype.Component;

import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.function.Supplier;

/**
 * Coordinates callbacks and generation changes for one worker address.
 *
 * <p>Status and cache callbacks use shared leases, so they do not block each
 * other. Generation retirement or replacement uses an exclusive lease and
 * therefore cannot race with an old callback's endpoint, request-lifecycle,
 * or cache side effects.</p>
 */
@Component
public class WorkerGenerationFence {

    private static final int STRIPES = 1024;
    private final Striped<ReadWriteLock> locks = Striped.readWriteLock(STRIPES);

    public <T> T read(String ipPort, Supplier<T> action) {
        return execute(locks.get(ipPort).readLock(), action);
    }

    public <T> T write(String ipPort, Supplier<T> action) {
        return execute(locks.get(ipPort).writeLock(), action);
    }

    private static <T> T execute(Lock lock, Supplier<T> action) {
        lock.lock();
        try {
            return action.get();
        } finally {
            lock.unlock();
        }
    }
}
