package org.flexlb.mockengine;

import static org.junit.jupiter.api.Assertions.*;
import java.util.*;
import java.nio.file.*;
import org.junit.jupiter.api.Test;
import com.fasterxml.jackson.databind.ObjectMapper;

class ClientConfigurationTest {
    static Map<String, Object> snapshot(Map<String, String> environment) {
        var c = JavaLoadClient.Config.fromEnvironment(environment);
        var p = new Playback(environment, c.loop, c.durationS);
        Map<String, Object> values = new LinkedHashMap<>();
        values.put("SEND_MODE", c.sendMode); values.put("SEND_MODE_QPS", c.sendModeQps);
        values.put("REPLAY_SPEED", c.replaySpeed); values.put("MAX_LAPS", p.maxLaps);
        values.put("LAP_IDENTITY", p.identity); values.put("RAMP_UP_SECONDS", c.rampUpSeconds);
        values.put("LOOP", c.loop); values.put("DURATION_S", c.durationS);
        values.put("MAX_CONCURRENCY", c.maxConcurrency); values.put("TIMEOUT_MS", c.timeoutMs);
        values.put("RESPONSE_TIMEOUT", c.responseTimeoutSeconds); values.put("N_CHANNELS", c.nChannels);
        values.put("EVENT_LOOP_THREADS", c.eventLoopThreads);
        values.put("FETCH_OUTPUT_STREAM", c.fetchOutputStream);
        values.put("REPLAY_UNIQUE_PREFIX", c.replayUniquePrefix);
        return values;
    }

    @Test void missingAndClearedValuesResolveFromSharedDefaults() throws Exception {
        Properties defaults = new Properties();
        try (var input = getClass().getResourceAsStream("/load-client-defaults.properties")) {
            assertNotNull(input); defaults.load(input);
        }
        Map<String, String> explicit = new HashMap<>(), cleared = new HashMap<>();
        defaults.forEach((key, value) -> { explicit.put(key.toString(), value.toString()); cleared.put(key.toString(), ""); });
        assertEquals(snapshot(explicit), snapshot(Map.of()));
        assertEquals(snapshot(explicit), snapshot(cleared));
        assertFalse((Boolean)snapshot(Map.of()).get("REPLAY_UNIQUE_PREFIX"));
    }

    @Test void invalidLimitsFailAtEnvironmentBoundary() {
        for (String key : List.of("MAX_CONCURRENCY", "TIMEOUT_MS", "RESPONSE_TIMEOUT", "N_CHANNELS", "EVENT_LOOP_THREADS"))
            assertThrows(IllegalArgumentException.class, () -> snapshot(Map.of(key, "0")));
        assertThrows(IllegalArgumentException.class, () -> snapshot(Map.of("REPLAY_SPEED", "NaN")));
        assertThrows(IllegalArgumentException.class, () -> snapshot(Map.of("TIMEOUT_MS", "200000")));
        assertThrows(IllegalArgumentException.class, () -> snapshot(Map.of("SEND_MODE", "uniform")));
    }

    /** Test-only process probe; validates Python projections against Java's actual parser. */
    public static void main(String[] args) throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        Map<String, String> environment = mapper.readValue(Files.readString(Path.of(args[0])),
                new com.fasterxml.jackson.core.type.TypeReference<Map<String,String>>() {});
        Files.writeString(Path.of(args[1]), mapper.writeValueAsString(snapshot(environment)));
    }
}
