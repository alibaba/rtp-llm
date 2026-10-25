package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import static org.junit.jupiter.api.Assertions.*;

/** Exercise the Python launch projection against the actual JVM admission contract. */
class MockRegisteredPresetsTest {
    @TempDir Path directory;

    @Test void everyRegisteredPresetPassesEngineAdmission() throws Exception {
        var output = directory.resolve("presets.json");
        var process = new ProcessBuilder(System.getenv().getOrDefault("PYTHON_BIN", "python3"), "-c", """
                import json, sys
                from pathlib import Path
                root = Path('../tools/online_eval').resolve()
                sys.path[:0] = [str(root / 'src'), str(root)]
                from adapters.perf_presets import load_preset_bundle, preset_names
                presets = {}
                for name in preset_names():
                    calibration = load_preset_bundle(name).performance
                    presets[name] = dict(engine=calibration.pop('engine_config'), calibration=calibration)
                print(json.dumps(presets))
                """).redirectErrorStream(true).redirectOutput(output.toFile()).start();
        boolean exited = process.waitFor(15, TimeUnit.SECONDS);
        if (!exited) process.destroyForcibly();
        assertTrue(exited, "Python preset projection timed out");
        var text = Files.readString(output);
        assertEquals(0, process.exitValue(), text);
        var presets = new ObjectMapper().readTree(text);
        assertTrue(presets.isObject() && !presets.isEmpty());
        for (var entries = presets.fields(); entries.hasNext();) {
            var entry = entries.next();
            assertDoesNotThrow(() -> {
                MockEngineConfig.parse(entry.getValue().path("engine"));
                MockPerformanceModel.loadCalibration(entry.getValue().path("calibration"));
            }, entry.getKey());
        }
    }
}
