package org.flexlb.mockengine;

import java.util.Map;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ClientEnvironmentTest {
    @Test void clearedAndMissingValuesSharePackagedDefaultsAndExplicitValuesWin() {
        var missing = new ClientEnvironment(Map.of());
        var cleared = new ClientEnvironment(Map.of("REPLAY_SPEED", ""));
        assertEquals(missing.number("REPLAY_SPEED"), cleared.number("REPLAY_SPEED"));
        assertEquals(2.5, new ClientEnvironment(Map.of("REPLAY_SPEED", "2.5")).number("REPLAY_SPEED"));
        assertTrue(missing.flag("FETCH_OUTPUT_STREAM"));
        assertThrows(IllegalArgumentException.class, () -> missing.text("UNKNOWN_DEFAULT"));
        assertThrows(IllegalArgumentException.class, () -> new ClientEnvironment(Map.of("FETCH_OUTPUT_STREAM", "maybe")).flag("FETCH_OUTPUT_STREAM"));
    }
}
