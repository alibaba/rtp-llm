package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/** Sampling contracts, independent of wall-clock scheduling. */
class BoundedPerformanceNoiseTest {
    private MockTiming timing(String json) throws Exception { return new MockTiming(new ObjectMapper().readTree(json)); }
    @Test void seedAndEngineScopeReplayDespiteOtherEngineActivity() throws Exception {
        var model = timing("{\"timing\":{\"seed\":42},\"forward\":{\"decode\":{\"jitter\":{\"stddev_ratio\":0.1,\"max_deviation_ms\":20}}}}");
        var a = model.forEngine("a"); var replay = model.forEngine("a"); var b = model.forEngine("b");
        boolean different = false;
        for (int i=0;i<100;i++) {
            var draw = a.sample("forward.decode",100);
            different |= draw.nanos() != b.sample("forward.decode",100).nanos();
            b.sample("forward.prefill",100);
            assertEquals(draw,replay.sample("forward.decode",100));
            assertTrue(Math.abs(draw.jitterMs()) <= 20);
        }
        assertTrue(different);
    }
    @Test void scaleIncludesJitterAndPauseAndRetainsSubMillisecondPrecision() throws Exception {
        var full = timing("{\"timing\":{\"seed\":7},\"forward\":{\"prefill\":{\"jitter\":{\"stddev_ms\":2,\"max_deviation_ms\":6},\"pause\":{\"probability\":1,\"duration_ms\":20}}}}");
        var fast = timing("{\"timing\":{\"seed\":7,\"time_scale\":0.001},\"forward\":{\"prefill\":{\"jitter\":{\"stddev_ms\":2,\"max_deviation_ms\":6},\"pause\":{\"probability\":1,\"duration_ms\":20}}}}");
        for (int i=0;i<100;i++) {
            var a=full.sample("forward.prefill",100); var b=fast.sample("forward.prefill",100);
            assertEquals(a.nanos()*0.001,b.nanos(),1);
            assertEquals(0.02,b.pauseMs(),1e-9);
            assertTrue(b.nanos()<1_000_000);
        }
    }
    @Test void pauseIsPerEventAndZeroProbabilityNeverPauses() throws Exception {
        var slow=timing("{\"rpc\":{\"get_worker_status\":{\"response\":{\"delay_ms\":2,\"pause\":{\"probability\":1,\"duration_ms\":50}}}}}");
        assertEquals(52_000_000,slow.sample("rpc.get_worker_status.response",0).nanos());
        assertEquals(0,slow.sample("rpc.get_cache_status.response",0).nanos());
    }
    @Test void strictSchemaRejectsLegacyAndMisspellings() throws Exception {
        var mapper=new ObjectMapper();
        for(String json:new String[]{"{\"sleep_scale\":1}","{\"jitter_pct\":0}","{\"prefill\":{}}", "{\"mock\":{\"forward\":{\"prefill\":{\"noise\":{}}}}}"})
            assertThrows(IllegalArgumentException.class,()->MockPerformanceModel.loadCalibration(mapper.readTree(json)));
        for(String json:new String[]{"{\"timing\":{\"seed\":1.5}}", "{\"timing\":{\"time_scale\":0}}", "{\"forward\":{\"prefill\":{\"jitter\":{\"stddev_mss\":1}}}}", "{\"rpc\":{\"typo\":{}}}"})
            assertThrows(IllegalArgumentException.class,()->timing(json));
    }
    @Test void runtimePatchUsesSameNestedSchemaAndUpdatesExistingTimingOwner() throws Exception {
        var mapper=new ObjectMapper(); var model=MockPerformanceModel.loadCalibration(mapper.createObjectNode()); var owner=model.timing();
        model.applyOverrides(mapper.readTree("{\"mock\":{\"timing\":{\"time_scale\":0.1},\"forward\":{\"prefill\":{\"fixed_ms\":100,\"pause\":{\"probability\":1,\"duration_ms\":50}}}}}"));
        assertEquals(15_000_000,owner.sample("forward.prefill",100).nanos());
        assertThrows(IllegalArgumentException.class,()->model.applyOverrides(mapper.readTree("{\"prefill_fixed_ms\":100}")));
    }
}
