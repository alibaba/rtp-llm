package org.flexlb.service.lease;

import org.flexlb.config.PrefillLeaseConfig;
import org.flexlb.dao.master.PrefillTtftWindow;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class PrefillTtftStatusReaderTest {
    private final WorkerDirectory directory = mock(WorkerDirectory.class);
    private final PrefillLeaseConfig config = new PrefillLeaseConfig();
    private final PrefillTtftStatusReader reader = new PrefillTtftStatusReader(directory);

    @Test
    void weightsRequestsAndExcludesStandbyAndIdleWorkersWithoutZeroSamples() {
        var a = worker(100, 1);
        var b = worker(900, 3);
        var idle = worker(0, 0);
        var standby = mock(WorkerStatus.class);
        when(directory.statusSnapshot(RoleType.PREFILL))
                .thenReturn(Map.of("a", a, "b", b, "idle", idle, "standby", standby));
        var result = reader.read(config, List.of("a", "b", "idle"), 100_000, 10_000);
        assertEquals(250, result.meanMs());
        assertEquals(4, result.samples());
        verify(a.prefillTtftWindow()).read(40_000, 100_000, 30_000, 10_000);
        org.mockito.Mockito.verifyNoInteractions(standby);
    }

    @Test
    void missingOrStaleServingWorkerHoldsInsteadOfUsingPartialMean() {
        var a = worker(100, 1);
        var b = worker(900, 3);
        when(directory.statusSnapshot(RoleType.PREFILL)).thenReturn(Map.of("a", a, "b", b));
        assertThrows(IllegalStateException.class, () -> reader.read(config, List.of("a", "missing"), 100_000, 10_000));
        when(b.prefillTtftWindow().read(40_000, 100_000, 30_000, 10_000)).thenThrow(new IllegalStateException());
        assertThrows(IllegalStateException.class, () -> reader.read(config, List.of("a", "b"), 100_000, 10_000));
    }

    @Test
    void allIdleIsUnavailableRatherThanZeroTtft() {
        var idle = worker(0, 0);
        when(directory.statusSnapshot(RoleType.PREFILL)).thenReturn(Map.of("idle", idle));
        assertThrows(IllegalStateException.class, () -> reader.read(config, List.of("idle"), 100_000, 10_000));
    }

    private WorkerStatus worker(double total, long count) {
        var worker = mock(WorkerStatus.class);
        var window = mock(PrefillTtftWindow.class);
        when(worker.isActiveGeneration()).thenReturn(true);
        when(worker.isAlive()).thenReturn(true);
        when(worker.prefillTtftWindow()).thenReturn(window);
        when(window.read(40_000, 100_000, 30_000, 10_000))
                .thenReturn(new PrefillTtftWindow.Measurement(total, count, 99_000));
        return worker;
    }
}
