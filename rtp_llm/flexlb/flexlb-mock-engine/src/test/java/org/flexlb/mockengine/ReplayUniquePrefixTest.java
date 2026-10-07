package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.Path;
import java.util.Collections;
import static org.junit.jupiter.api.Assertions.*;

/** Loop identity now transforms actual tokens and every block key. */
class ReplayUniquePrefixTest {
    @TempDir Path tempDir;
    private JavaLoadClient client() {
        JavaLoadClient.Config config = new JavaLoadClient.Config(
            "trace.jsonl", "127.0.0.1:7001", "127.0.0.1:7003",
            0, 16, 10.0, 1, tempDir.resolve("out").toString(), 1, 0, 0,
            120_000L, 500.0, false, true, 1, 1, 0L, 120, true,
            "engine_service", "", false, 10, 1000, 0, 0, "", false, "", true);
        return new JavaLoadClient(config);
    }
    @Test void loopTransformsTokensKeysAndIterationTogether() {
        var tokens=Collections.nCopies(2048,7);
        var keys=JavaLoadClient.computeBlockKeys(tokens,512);
        var record=new JavaLoadClient.TraceRecord(1,"r","trace",0,2048,8,keys,tokens,50,512);
        var client=client();
        var first=client.makeLoopRequest(record,1,0);
        var again=client.makeLoopRequest(record,1,0);
        var second=client.makeLoopRequest(record,2,0);
        assertEquals(first.blockKeys,again.blockKeys);
        assertTrue(Collections.disjoint(keys,first.blockKeys));
        assertTrue(Collections.disjoint(first.blockKeys,second.blockKeys));
        assertEquals(JavaLoadClient.computeBlockKeys(first.tokenIds,512),first.blockKeys);
        assertEquals(1,first.iteration);assertEquals(512,first.cacheKeyBlockSize);
        assertNotEquals(record.sourceRid,first.sourceRid);
        assertEquals(7,record.tokenIds.get(0));
    }
}
