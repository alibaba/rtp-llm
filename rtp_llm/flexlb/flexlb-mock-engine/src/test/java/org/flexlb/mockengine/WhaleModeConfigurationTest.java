package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

class WhaleModeConfigurationTest {
    @TempDir Path directory;

    private JavaMockEngineCluster.Config config(String... options) {
        List<String> args = new ArrayList<>(List.of("--endpoint-file", "endpoints.json",
                "--performance", "performance.json", "--master-config", "master.json"));
        args.addAll(List.of(options));
        return JavaMockEngineCluster.Config.parse(args.toArray(String[]::new));
    }

    @Test
    void defaultModeRetainsLocalDiscoveryAndNoPrivateMonitor() {
        var config = config();
        assertFalse(config.whale);
        assertFalse(config.kmonitor);
        assertNotNull(config.discoveryFile);
    }

    @Test
    void whaleRequiresOneEngineAndNeverGeneratesLocalDiscovery() {
        assertThrows(IllegalArgumentException.class, () -> config("--whale", "true"));
        var config = config("--whale", "true", "--n-prefill", "1", "--n-decode", "0",
                "--host", "10.1.2.3");
        assertNull(config.discoveryFile);
        assertEquals("10.1.2.3", JavaMockEngineCluster.declaredHost(config, 0));
        assertThrows(IllegalArgumentException.class, () -> config("--kmonitor", "true"));
    }

    @Test
    void wrapperRefusesImplicitActivationAndMapsWhalePortsAndFetch() throws Exception {
        Path script = Path.of("whale/start.sh").toAbsolutePath();
        ProcessBuilder refused = new ProcessBuilder("sh", script.toString()).redirectErrorStream(true);
        refused.environment().remove("FLEXLB_MOCK_WHALE");
        Process first = refused.start();
        assertTrue(first.waitFor(3, TimeUnit.SECONDS));
        assertEquals(2, first.exitValue());

        Path javaExecutable = directory.resolve("java");
        Files.writeString(javaExecutable, "#!/bin/sh\nprintf '%s\\n' \"$@\"\n");
        assertTrue(javaExecutable.toFile().setExecutable(true));
        ProcessBuilder builder = new ProcessBuilder("sh", script.toString()).redirectErrorStream(true);
        var env = builder.environment();
        env.put("PATH", directory + ":" + env.get("PATH"));
        env.put("FLEXLB_MOCK_WHALE", "1");
        env.put("ROLE_TYPE", "DECODE");
        env.put("START_PORT", "22290");
        env.put("POD_IP", "10.1.2.3");
        env.put("FETCH_OUTPUT_STREAM", "0");
        env.put("MOCK_PERFORMANCE_CONFIG", "performance.json");
        env.put("MOCK_MASTER_CONFIG", "master.json");
        env.put("MOCK_RUN_DIR", directory.toString());
        env.remove("MOCK_KMONITOR_ENABLED");
        Process process = builder.start();
        assertTrue(process.waitFor(3, TimeUnit.SECONDS));
        String args = new String(process.getInputStream().readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
        assertEquals(0, process.exitValue(), args);
        assertTrue(args.contains("--n-prefill\n0\n--n-decode\n1\n"), args);
        assertTrue(args.contains("--base-grpc-port\n22291\n--auto-fetch\ntrue\n"), args);
        assertTrue(args.contains("--kmonitor\ntrue\n"), args);
    }
}
