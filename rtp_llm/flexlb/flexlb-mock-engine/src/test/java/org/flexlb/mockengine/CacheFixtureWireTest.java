package org.flexlb.mockengine;

import io.grpc.Drainable;
import io.grpc.KnownLength;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import java.io.ByteArrayOutputStream;
import java.util.concurrent.atomic.AtomicLong;
import static org.junit.jupiter.api.Assertions.*;

class CacheFixtureWireTest {
    @org.junit.jupiter.api.io.TempDir java.nio.file.Path directory;
    private static long hash(long x) {
        x = (x ^ (x >>> 30)) * 0xbf58476d1ce4e5b9L;
        x = (x ^ (x >>> 27)) * 0x94d049bb133111ebL;
        return x ^ (x >>> 31);
    }
    private static EngineRpcService.CacheStatusPB reference(int worker, int workers, int unique,
                                                            int shared, long offset, long version) {
        var b = EngineRpcService.CacheStatusPB.newBuilder().setVersion(version);
        for (int i = 0; i < unique; i++) b.putCacheKeys(hash(((long)worker << 40) + offset + i), true);
        for (int segment : new int[]{worker, (worker + workers - 1) % workers})
            for (int i = 0; i < shared; i++) b.putCacheKeys(hash(Long.MIN_VALUE + ((long)segment << 40) + i), true);
        return b.build();
    }
    @Test void canonicalProtobufContentAndBytesMatchAcrossGeometryAndOffsets() throws Exception {
        for (int worker : new int[]{0, 1, 119}) for (int shared : new int[]{0, 1, 65})
            for (long offset : new long[]{0, 117, 99991}) {
                var keys = CacheFixtureWire.Keys.create(worker, 120, 176, shared, offset, 129);
                var header = EngineRpcService.CacheStatusPB.newBuilder().setVersion(129).build();
                var frame = new CacheFixtureWire.Frame(header, keys);
                var expected = reference(worker, 120, 176, shared, offset, 129);
                assertEquals(expected, frame.toProto());
                try (var stream = frame.stream()) {
                    assertInstanceOf(KnownLength.class, stream);
                    assertEquals(expected.getSerializedSize(), stream.available());
                    assertArrayEquals(expected.toByteArray(), stream.readAllBytes());
                    assertEquals(0, stream.available());
                    assertEquals(-1, stream.read());
                }
            }
    }
    @Test void partialReadsDrainAndScalarMetadataUseStandardClientParser() throws Exception {
        var header = EngineRpcService.CacheStatusPB.newBuilder().setVersion(7).setAvailableKvCache(123)
                .setTotalKvCache(456).setBlockSize(1024).build();
        var frame = new CacheFixtureWire.Frame(header, CacheFixtureWire.Keys.create(0, 120, 176, 65, 0, 7));
        var expected = reference(0, 120, 176, 65, 0, 7).toBuilder().mergeFrom(header).build();
        try (var in = frame.stream()) {
            var out = new ByteArrayOutputStream(); byte[] b = new byte[7]; int n;
            assertEquals(0, in.read(b, 0, 0));
            while ((n = in.read(b)) >= 0) out.write(b, 0, n);
            assertEquals(expected, EngineRpcService.CacheStatusPB.parseFrom(out.toByteArray()));
        }
        try (var in = frame.stream()) {
            var out = new ByteArrayOutputStream(); assertEquals(expected.getSerializedSize(), ((Drainable)in).drainTo(out));
            assertEquals(expected, EngineRpcService.CacheStatusPB.parseFrom(out.toByteArray()));
        }
    }
    @Test void inFlightWireSnapshotSurvivesChurnAndSameVersionRemainsFull() {
        var clock = new AtomicLong();
        var f = new CacheSyncFixture(0, 120, 176, 65, 2, 1000, clock::get);
        var first = f.wireResponse(-1); f.churn(17, 1); clock.set(5001);
        var second = f.wireResponse(2);
        assertEquals(reference(0, 120, 176, 65, 0, 2), first.toProto());
        assertEquals(reference(0, 120, 176, 65, 85, 7), second.toProto());
        assertEquals(306, f.wireResponse(7).toProto().getCacheKeysCount());
        assertEquals(306, f.wireResponse(8).toProto().getCacheKeysCount());
        assertEquals(1224L, f.snapshot().get("entries"));
        assertEquals(85L, f.snapshot().get("removed_keys"));
    }
    @Test void inFlightSnapshotSurvivesGrowthAndLegacySameVersionStillReturnsKeys() {
        var fixture = new CacheSyncFixture(1, 3, 1, 0, 1, 0, () -> 0L);
        var beforeGrowth = fixture.wireResponse(-1);
        fixture.grow(8, 2);
        var afterGrowth = fixture.wireResponse(fixture.version());
        assertEquals(reference(1, 3, 1, 0, 0, 1), beforeGrowth.toProto());
        assertEquals(reference(1, 3, 8, 2, 0, 2), afterGrowth.toProto());
        assertEquals(12, fixture.snapshot().get("keys"));
        assertEquals(11L, fixture.snapshot().get("added_keys"));
        assertEquals(afterGrowth.toProto(), fixture.response(fixture.version()));
    }

    @Test void productionSizeResponseMatchesReference() {
        var keys = CacheFixtureWire.Keys.create(119, 120, 176000, 65000, 117, 4);
        var frame = new CacheFixtureWire.Frame(EngineRpcService.CacheStatusPB.newBuilder().setVersion(4).build(), keys);
        assertEquals(reference(119, 120, 176000, 65000, 117, 4), frame.toProto());
    }

    @Test void unmodifiedGrpcClientDecodesFixtureAndOrdinaryResponses() throws Exception {
        var perf = directory.resolve("perf.json"); java.nio.file.Files.writeString(perf, "{\"block_size\":2}");
        var config = directory.resolve("master.json"); MockMasterConfig.writeWithPrefillExpression(config, "20");
        var model = MockPerformanceModel.load(perf.toString(), config.toString());
        try (var cluster = MockEngineTestCluster.create(model, 61580, 1, 0)) {
            var service = cluster.prefill(0);
            var server = io.grpc.netty.NettyServerBuilder.forPort(0).addService(CacheFixtureWire.bind(service)).build().start();
            var channel = io.grpc.ManagedChannelBuilder.forAddress("127.0.0.1", server.getPort()).usePlaintext().build();
            try {
                var stub = org.flexlb.engine.grpc.RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3, java.util.concurrent.TimeUnit.SECONDS);
                var request = EngineRpcService.CacheVersionPB.newBuilder().setNeedCacheKeys(true).setLatestCacheVersion(-1).build();
                assertEquals(0, stub.getCacheStatus(request).getCacheKeysCount());
                assertTrue(stub.getWorkerStatus(EngineRpcService.StatusVersionPB.getDefaultInstance()).getAlive());
                service.cacheFixture("seed", 0, 120, 176, 65, 0, 0, 0);
                var response = stub.getCacheStatus(request);
                var expected = reference(0, 120, 176, 65, 0, response.getVersion());
                assertEquals(expected.getCacheKeysMap(), response.getCacheKeysMap());
                assertEquals(2, response.getBlockSize());
                assertTrue(response.getTotalKvCache() > 0);
                assertEquals(0, stub.getCacheStatus(request.toBuilder().setNeedCacheKeys(false).build()).getCacheKeysCount());
            } finally {
                channel.shutdownNow().awaitTermination(3, java.util.concurrent.TimeUnit.SECONDS);
                server.shutdownNow().awaitTermination(3, java.util.concurrent.TimeUnit.SECONDS);
            }
        }
    }
}
