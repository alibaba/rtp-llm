package org.flexlb.dao.master;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PrefillChunkLeaseTest {
    @Test
    void requiresMatchingConfirmationAndChunkAfterActivation() {
        PrefillChunkLease lease = new PrefillChunkLease();
        assertTrue(lease.allows(chunk(0)));
        lease.change(true);
        long version = lease.command().version();
        assertFalse(lease.allows(chunk(version)));
        lease.confirm(version, true);
        assertTrue(lease.allows(chunk(version)));
        assertFalse(lease.allows(chunk(0)));
        lease.confirm(0, false);
        assertFalse(lease.allows(chunk(version)));
    }

    @Test
    void revokeImmediatelyBlocksEvenBeforeWorkerAcknowledges() {
        PrefillChunkLease lease = new PrefillChunkLease();
        lease.change(true);
        long first = lease.command().version();
        lease.confirm(first, true);
        assertTrue(lease.allows(chunk(first)));
        lease.change(false);
        assertFalse(lease.allows(chunk(first)));
        lease.confirm(first, true);
        assertFalse(lease.allows(chunk(first)));
        lease.change(true);
        long next = lease.command().version();
        assertTrue(next > first);
        lease.confirm(next, true);
        assertFalse(lease.allows(chunk(first)));
        assertTrue(lease.allows(chunk(next)));
        assertFalse(lease.change(true));
    }

    private static PrefillChunk chunk(long version) {
        return new PrefillChunk("engine", 1, 1024, 1, Long.MAX_VALUE, false, version);
    }
}
