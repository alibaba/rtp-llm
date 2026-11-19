package org.flexlb.dao.master;

/**
 * Explicit, process-local permission to publish new chunks. Access requires the worker lock.
 */
public final class PrefillChunkLease {
    private static final java.util.concurrent.atomic.AtomicLong NEXT_VERSION =
            new java.util.concurrent.atomic.AtomicLong(System.currentTimeMillis());
    private long version;
    private boolean enabled;
    private long confirmedVersion;
    private boolean confirmedEnabled;

    public record Command(long version, boolean enabled) {
    }

    public Command command() {
        return version == 0 ? null : new Command(version, enabled);
    }

    public boolean change(boolean nextEnabled) {
        if (version != 0 && enabled == nextEnabled) {
            return false;
        }
        version = NEXT_VERSION.incrementAndGet();
        enabled = nextEnabled;
        return true;
    }

    public void confirm(long reportedVersion, boolean reportedEnabled) {
        // A response without an acknowledgement invalidates an old confirmation after an engine restart.
        confirmedVersion = reportedVersion;
        confirmedEnabled = reportedEnabled;
    }

    public boolean allows(PrefillChunk chunk) {
        return version == 0 || enabled && confirmedVersion == version && confirmedEnabled && chunk != null
                && chunk.leaseVersion() == version;
    }

    public boolean serving() {
        return version > 0 && enabled && confirmedVersion == version && confirmedEnabled;
    }

    public boolean confirmed() {
        return version > 0 && confirmedVersion == version && confirmedEnabled == enabled;
    }
}
