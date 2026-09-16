package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class WhaleEosModelTest {
    private static final ObjectMapper JSON = new ObjectMapper();
    private MockEosModel model(String json) throws Exception {
        return MockEosModel.load(JSON.readTree(json));
    }
    private EngineRpcService.GenerateInputPB request(long id, int cap, int min, boolean ignore) {
        return EngineRpcService.GenerateInputPB.newBuilder().setRequestId(id)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(cap).setMinNewTokens(min).setIgnoreEos(ignore)).build();
    }
    @Test void disabledIsExactlyLegacyIncludingExplicitReplayLengths() throws Exception {
        for (String config : new String[]{"{}", "{\"enabled\":false}"}) {
            var model = model(config);
            assertEquals(393216, model.outputLength(request(1, 393216, 0, false), 393216, false));
            assertEquals(37, model.outputLength(request(1, 10, 0, true), 37, true));
        }
    }
    @Test void honorsRealEngineMinimumMaximumAndIgnoreEos() throws Exception {
        var model = model("{\"enabled\":true,\"distribution\":\"geometric\",\"mean_tokens\":1,\"seed\":42}");
        assertEquals(1, model.outputLength(request(1, 393216, 0, false), 393216, false));
        assertEquals(128, model.outputLength(request(1, 393216, 128, false), 393216, false));
        assertEquals(32, model.outputLength(request(1, 32, 128, false), 32, false));
        assertEquals(393216, model.outputLength(request(1, 393216, 0, true), 393216, false));
        assertEquals(37, model.outputLength(request(1, 100, 0, false), 37, true));
        assertEquals(10, model.outputLength(request(1, 10, 0, false), 37, true));
    }
    @Test void independentEnginesAndRetriesAgreeAndDistributionHasATail() throws Exception {
        String config = "{\"enabled\":true,\"distribution\":\"geometric\",\"mean_tokens\":1024,\"seed\":17}";
        var a = model(config); var b = model(config);
        long sum = 0; int tail = 0;
        for (int i = 0; i < 20000; i++) {
            var request = request(i, 393216, 0, false);
            int length = a.outputLength(request, 393216, false);
            assertEquals(length, b.outputLength(request, 393216, false));
            assertTrue(length >= 1 && length <= 393216);
            sum += length;
            if (length > 4096) tail++;
        }
        assertTrue(sum / 20000.0 > 990 && sum / 20000.0 < 1060);
        assertTrue(tail > 250 && tail < 500, "retain a long tail, not a constant truncation");
    }
    @Test void enabledModelRejectsMissingOrInvalidParameters() {
        for (String config : new String[]{"{\"enabled\":true}",
                "{\"enabled\":true,\"distribution\":\"geometric\",\"mean_tokens\":0,\"seed\":1}",
                "{\"enabled\":true,\"distribution\":\"geometric\",\"mean_tokens\":100}"}) {
            assertThrows(IllegalArgumentException.class, () -> model(config));
        }
    }
}
