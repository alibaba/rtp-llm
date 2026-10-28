package org.flexlb.mock.engine.rtp;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.grpc.*;
import io.grpc.stub.StreamObserver;
import java.nio.file.*;
import java.net.*;
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
    private final Set<Integer> allocated = new HashSet<>();
    private int port() throws Exception {
        while (true) try (var socket = new ServerSocket(0, 1, InetAddress.getByName("127.0.0.1"))) {
            if (allocated.add(socket.getLocalPort())) return socket.getLocalPort();
        }
    }
    private EngineHostApi.Create create(String id) throws Exception {
        return new EngineHostApi.Create(id, port(), port());
    }
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
            var p = host.create(create("prefill-0"));
            var d = host.create(create("decode-0"));
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
            assertThrows(IllegalArgumentException.class, () -> host.create(create("prefill-0")));
            var fresh = host.create(create("prefill-1"));
            assertNotEquals(p.rpc(), fresh.rpc());
            assertNotEquals(p.control(), fresh.control());
            assertEquals(2, host.status(new None()).size());
        }
    }
    @Test void invalidConfigurationAndUnknownStopDoNotAffectExistingInstances() throws Exception {
        config("decode-0", "decode"); config("prefill-0", "decode");
        try (var host = new EngineHost(directory)) {
            var d = host.create(create("decode-0"));
            assertThrows(IllegalArgumentException.class, () -> host.create(create("prefill-0")));
            assertThrows(IllegalArgumentException.class, () -> host.stop(new EngineHostApi.Stop("prefill-99")));
            assertEquals(List.of(d), host.status(new None()));
            assertThrows(IllegalArgumentException.class, () -> create("../outside"));
        }
    }
    @Test void retiredListenersCannotBeReassignedAcrossProtocols() throws Exception {
        config("prefill-0", "prefill"); config("prefill-1", "prefill"); config("prefill-2", "prefill");
        try (var host = new EngineHost(directory)) {
            var original = create("prefill-0");
            var node = host.create(original);
            host.stop(new EngineHostApi.Stop(node.id()));
            assertThrows(IllegalArgumentException.class, () -> host.create(
                new EngineHostApi.Create("prefill-1", original.controlPort(), port())));
            assertThrows(IllegalArgumentException.class, () -> host.create(
                new EngineHostApi.Create("prefill-2", port(), original.rpcPort())));
            assertTrue(host.status(new None()).isEmpty());
            for (int port : List.of(original.rpcPort(), original.controlPort())) {
                try (var socket = new Socket()) {
                    assertThrows(java.io.IOException.class, () -> socket.connect(new InetSocketAddress("127.0.0.1", port), 500));
                }
            }
        }
    }
    @Test void anotherHostCannotOverwriteRetiredArtifacts() throws Exception {
        config("prefill-0", "prefill");
        try (var first = new EngineHost(directory)) { first.create(create("prefill-0")); }
        var old = Files.readAllBytes(directory.resolve("prefill-0/endpoints.json"));
        try (var replacement = new EngineHost(directory)) {
            assertThrows(FileAlreadyExistsException.class, () -> replacement.create(create("prefill-0")));
            assertArrayEquals(old, Files.readAllBytes(directory.resolve("prefill-0/endpoints.json")));
            assertTrue(replacement.status(new None()).isEmpty());
        }
    }
    @Test void listenerBindFailureDoesNotSelectADifferentPortAndReleasesPartialStartup() throws Exception {
        config("prefill-0", "prefill");
        try (var host = new EngineHost(directory); var busy = new ServerSocket(0, 1, InetAddress.getByName("127.0.0.1"))) {
            int rpcPort = port();
            assertThrows(java.io.IOException.class, () -> host.create(new EngineHostApi.Create("prefill-0", rpcPort, busy.getLocalPort())));
            assertTrue(host.status(new None()).isEmpty());
            assertFalse(Files.exists(directory.resolve("prefill-0/ready.json")));
            try (var available = new ServerSocket()) {
                available.bind(new InetSocketAddress("127.0.0.1", rpcPort));
            }
        }
    }
    @Test void invalidPortsAreRejectedBeforeCreatingAnInstance() {
        for (int port : List.of(-1, 0, 65536)) {
            assertThrows(IllegalArgumentException.class, () -> new EngineHostApi.Create("prefill-0", port, 12345));
            assertThrows(IllegalArgumentException.class, () -> new EngineHostApi.Create("prefill-0", 12345, port));
        }
        assertThrows(IllegalArgumentException.class, () -> new EngineHostApi.Create("prefill-0", 12345, 12345));
    }

}
