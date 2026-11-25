package org.flexlb.service.monitor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TtftRequestIdsTest {
    @Test
    void keepsCallAndAttemptNamespacesAndRemovesOnlyTheInnerPrefillTransportSuffix() {
        String call = "opaque#attempt9#call0123456789abcdef0123456789abcdef";
        for (String id : new String[]{call, call + "#attempt1", "a-n1" + call, "a-p2" + call,
                "opaque-p-0123abcd" + call}) {
            assertEquals(id, TtftRequestIds.dispatchId(id));
            assertEquals(id, TtftRequestIds.dispatchId(id + "-p-0123abcd"));
            assertEquals(id,
                    TtftRequestIds.dispatchId(id + "-p-0123abcd#call0123456789abcdef0123456789abcdef#attempt1"));
            assertEquals(id + "-0123abcd", TtftRequestIds.dispatchId(id + "-0123abcd"));
        }
    }
}
