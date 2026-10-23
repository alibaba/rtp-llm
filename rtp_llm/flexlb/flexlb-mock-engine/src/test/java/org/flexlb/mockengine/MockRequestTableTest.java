package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import static org.junit.jupiter.api.Assertions.*;

class MockRequestTableTest {
    @Test void finishingDeadlineRetainsLateFetchAndDiagnosticState() {
        var requests = new MockRequestTable();
        var output = new MockResponseQueue();
        var deadline = new MockRequestDeadline(1000);
        requests.output(1, output);
        requests.state(1, "waiting_fetch");
        assertNull(requests.inheritDeadline(1, deadline));
        assertSame(deadline, requests.inheritDeadline(1, new MockRequestDeadline(500)));
        requests.finish(1);
        assertNull(requests.deadline(1));
        assertSame(output, requests.output(1));
        assertEquals("waiting_fetch", requests.state(1));
        requests.removeOutput(1);
        requests.removeStatesIf((id, state) -> true);
        assertEquals(0, requests.size());
    }

    @Test void staleCleanupCannotRemoveReplacementOutputOrState() {
        var requests = new MockRequestTable();
        var old = new MockResponseQueue();
        var replacement = new MockResponseQueue();
        requests.output(1, old);
        requests.state(1, "completed");
        requests.removeOutputsIf((id, queue) -> { requests.output(id, replacement); return true; });
        requests.removeStatesIf((id, state) -> { requests.state(id, "received"); return true; });
        assertSame(replacement, requests.output(1));
        assertEquals("received", requests.state(1));
    }

    @Test @Timeout(10)
    void concurrentRegistrationAndPruningKeepLiveComponents() throws Exception {
        var requests = new MockRequestTable();
        var executor = java.util.concurrent.Executors.newFixedThreadPool(2);
        try {
            var writer = executor.submit(() -> {
                for (long id = 0; id < 1000; id++) {
                    requests.ensureOutput(id, MockResponseQueue::new);
                    requests.state(id, "received");
                }
            });
            var cleaner = executor.submit(() -> {
                for (int i = 0; i < 100; i++) requests.removeStatesIf((id, state) -> state.equals("completed"));
            });
            writer.get(); cleaner.get();
            assertEquals(1000, requests.outputCount());
            assertEquals(1000, requests.states().size());
            requests.clearOutputs(); requests.clearStates(); requests.closeDeadlines();
            assertEquals(0, requests.size());
        } finally { executor.shutdownNow(); }
    }
}
