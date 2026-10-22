package org.flexlb.mockengine;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ShutdownTasksTest {
    @Test void everyOwnerIsAttemptedAndEveryFailureRemainsVisible() {
        List<String> attempted = new ArrayList<>();
        var error = assertThrows(IllegalStateException.class, () -> ShutdownTasks.runAll(List.of(
                () -> { attempted.add("first"); throw new IllegalStateException("one"); },
                () -> { attempted.add("second"); throw new IllegalArgumentException("two"); },
                () -> attempted.add("third"))));
        assertEquals(List.of("first", "second", "third"), attempted);
        assertEquals(2, error.getSuppressed().length);
    }
}
