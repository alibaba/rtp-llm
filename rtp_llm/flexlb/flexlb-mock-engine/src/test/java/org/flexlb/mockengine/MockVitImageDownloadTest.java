package org.flexlb.mockengine;

import com.sun.net.httpserver.HttpServer;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import java.net.InetSocketAddress;
import java.util.Map;
import java.util.concurrent.*;
import static org.junit.jupiter.api.Assertions.*;

class MockVitImageDownloadTest {
    @Test @Timeout(10)
    void responseBodyMustRespectTheDownloadDeadlineAfterHeadersArrive() throws Exception {
        var origin = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        origin.createContext("/image", exchange -> {
            try {
                exchange.sendResponseHeaders(200, 0);
                exchange.getResponseBody().write(1);
                exchange.getResponseBody().flush();
                entered.countDown();
                release.await();
            } catch (InterruptedException error) { Thread.currentThread().interrupt(); }
            finally { exchange.close(); }
        });
        origin.start();
        try (var workers = Executors.newVirtualThreadPerTaskExecutor()) {
            var input = EngineRpcService.MultimodalInputPB.newBuilder().setMultimodalType(1)
                    .setMultimodalUrl("http://127.0.0.1:" + origin.getAddress().getPort() + "/image").build();
            var prepared = workers.submit(() -> MockVitImage.prepare(input, 8, 1024, 1 << 20, 1000, Map.of()));
            try {
                assertTrue(entered.await(2, TimeUnit.SECONDS));
                var error = assertThrows(ExecutionException.class, () -> prepared.get(2, TimeUnit.SECONDS));
                assertInstanceOf(io.grpc.StatusRuntimeException.class, error.getCause());
                assertEquals(906, MockRpcErrors.downstream(error.getCause(), 0).getErrorCode());
            } finally { release.countDown(); }
        } finally { release.countDown(); origin.stop(0); }
    }
}
