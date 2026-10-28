package org.flexlb.mock.engine.rtp;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.grpc.*;
import io.grpc.stub.StreamObserver;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.mock.api.model.Types.None;
import org.flexlb.mock.engine.rtp.host.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class EngineHostTest {
    @TempDir Path directory;
    private void config(String id, String role) throws Exception {
        var json = new ObjectMapper(); ObjectNode value = json.valueToTree(RtpNodeTest.configuration(role));
        value.put("id", id);
        ((ObjectNode)value.path("component").path("config").path("compute")).put("expression", "1000");
        var path = directory.resolve(id); Files.createDirectory(path);
        json.writeValue(path.resolve("config.json").toFile(), value);
    }
    @Test void stoppingOneObjectPreservesOtherInstancesAndDoesNotReuseAnIdentity() throws Exception {
        config("prefill-0", "prefill"); config("decode-0", "decode"); config("prefill-1", "prefill");
        try (var host = new EngineHost(directory)) {
            var p = host.create(new EngineHostApi.Create("prefill-0", List.of()));
            var d = host.create(new EngineHostApi.Create("decode-0", List.of()));
            assertEquals(2, host.status(new None()).size());
            assertNotEquals(p.rpc(), d.rpc()); assertNotEquals(p.metrics(), d.metrics());
            assertTrue(Files.exists(directory.resolve("prefill-0/ready.json")));
            var channel = ManagedChannelBuilder.forTarget(p.rpc()).usePlaintext().build();
            try {
                var replies = new LinkedBlockingQueue<Object>();
                RpcServiceGrpc.newStub(channel).generateStreamCall(LocalPrefillTest.input(1, 5000, true), new StreamObserver<>() {
                    public void onNext(GenerateOutputsPB value) { replies.add(value); }
                    public void onError(Throwable error) { replies.add(error); }
                    public void onCompleted() { replies.add("complete"); }
                });
                var rpc = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(2, TimeUnit.SECONDS);
                long end = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
                while (rpc.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getRunningQueryLen() == 0) {
                    assertTrue(System.nanoTime() < end); Thread.sleep(5);
                }
                assertEquals(p, host.stop(new EngineHostApi.Stop(p.id())));
                assertInstanceOf(Throwable.class, replies.poll(2, TimeUnit.SECONDS));
            } finally { channel.shutdownNow().awaitTermination(2, TimeUnit.SECONDS); }
            assertEquals(List.of(d), host.status(new None()));
            assertFalse(Files.exists(directory.resolve("prefill-0/ready.json")));
            assertTrue(Files.exists(directory.resolve("prefill-0/endpoints.json")));
            assertThrows(IllegalArgumentException.class, () -> host.create(new EngineHostApi.Create("prefill-0", List.of())));
            var fresh = host.create(new EngineHostApi.Create("prefill-1", List.of(p.rpc(), d.rpc())));
            assertNotEquals(p.rpc(), fresh.rpc());
            assertEquals(2, host.status(new None()).size());
        }
    }
    @Test void invalidConfigurationAndUnknownStopDoNotAffectExistingInstances() throws Exception {
        config("decode-0", "decode"); config("prefill-0", "decode");
        try (var host = new EngineHost(directory)) {
            var d = host.create(new EngineHostApi.Create("decode-0", List.of()));
            assertThrows(IllegalArgumentException.class, () -> host.create(new EngineHostApi.Create("prefill-0", List.of())));
            assertThrows(IllegalArgumentException.class, () -> host.stop(new EngineHostApi.Stop("prefill-99")));
            assertEquals(List.of(d), host.status(new None()));
            assertThrows(IllegalArgumentException.class, () -> new EngineHostApi.Create("../outside", List.of()));
        }
    }
}
