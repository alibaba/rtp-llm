package org.flexlb.mockengine;

import io.grpc.Drainable;
import io.grpc.KnownLength;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
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
        for (int i = 0; i < shared; i++) b.putCacheKeys(hash(Long.MIN_VALUE + i), true);
        return b.build();
    }
    @Test void canonicalProtobufContentAndBytesMatchAcrossGeometryAndOffsets() throws Exception {
        for (int worker : new int[]{0, 1, 119}) for (int shared : new int[]{0, 1, 65})
            for (long offset : new long[]{0, 117, 99991}) {
                var keys = CacheFixtureWire.Keys.create(worker, 120, 176, shared, offset, 129);
                for (int i = 0; i < keys.count(); i++) assertTrue(keys.contains(keys.key(i)));
                assertFalse(keys.contains(hash(((long)worker << 40) + offset - 1)));
                assertFalse(keys.contains(hash(((long)worker << 40) + offset + 176)));
                assertFalse(keys.contains(hash(Long.MIN_VALUE + shared)));
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
        var f = new CacheSyncFixture(0, 120, 176, 65, 2, clock::get);
        var first = f.wireResponse(-1); f.churn(17, 1); clock.set(5_000_001);
        var second = f.wireResponse(2);
        assertEquals(reference(0, 120, 176, 65, 0, 2), first.toProto());
        assertEquals(reference(0, 120, 176, 65, 85, 7), second.toProto());
        assertEquals(241, f.wireResponse(7).toProto().getCacheKeysCount());
        assertEquals(241, f.wireResponse(8).toProto().getCacheKeysCount());
        assertEquals(964L, f.snapshot().get("entries"));
        assertEquals(85L, f.snapshot().get("removed_keys"));
    }
    @Test void productionSizeResponseMatchesReference() {
        var keys = CacheFixtureWire.Keys.create(119, 120, 176000, 65000, 117, 4);
        var frame = new CacheFixtureWire.Frame(EngineRpcService.CacheStatusPB.newBuilder().setVersion(4).build(), keys);
        assertEquals(reference(119, 120, 176000, 65000, 117, 4), frame.toProto());
    }

    @Test void compactNaturalUnionUsesCanonicalWireAndIndependentStreams() throws Exception {
        var external = CacheFixtureWire.Keys.create(0, 1, 2, 2, 0, 3);
        long[] device = {Long.MIN_VALUE, -1, 0, Long.MAX_VALUE, external.key(0)};
        long[] host = {0, -1, 42, external.key(2)};
        var natural = CacheFixtureWire.NaturalKeys.takeSnapshots(device, host, external);
        var header = EngineRpcService.CacheStatusPB.newBuilder().setVersion(9).setBlockSize(512).build();
        var frame = new CacheFixtureWire.Frame(header, external, natural);
        var expected = reference(0, 1, 2, 2, 0, 9).toBuilder().mergeFrom(header);
        for (long key : new long[]{Long.MIN_VALUE, -1, 0, Long.MAX_VALUE, 42}) expected.putCacheKeys(key, true);
        var proto = expected.build();
        assertEquals(0, frame.header().getCacheKeysCount());
        assertEquals(proto, frame.toProto());
        try (var first = frame.stream(); var second = frame.stream()) {
            int head = first.read();
            assertEquals(proto, EngineRpcService.CacheStatusPB.parseFrom(second));
            var bytes = new ByteArrayOutputStream(); bytes.write(head);
            ((Drainable)first).drainTo(bytes);
            assertEquals(proto, EngineRpcService.CacheStatusPB.parseFrom(bytes.toByteArray()));
            assertEquals(proto.getSerializedSize(), bytes.size());
        }
    }

    @Test void compactNaturalEncodingAvoidsPerKeyProtobufAllocations() throws Exception {
        var bean = java.lang.management.ManagementFactory.getThreadMXBean();
        org.junit.jupiter.api.Assumptions.assumeTrue(bean instanceof com.sun.management.ThreadMXBean);
        var allocations = (com.sun.management.ThreadMXBean) bean;
        org.junit.jupiter.api.Assumptions.assumeTrue(allocations.isThreadAllocatedMemorySupported());
        allocations.setThreadAllocatedMemoryEnabled(true);
        long thread = Thread.currentThread().threadId();
        int count = 50_000;
        long[] keys = new long[count];
        for (int i = 0; i < count; i++) keys[i] = hash(i);
        var metadata = EngineRpcService.CacheStatusPB.newBuilder().setVersion(7).build();
        // Warm both serializers without including reference construction in compact's measurement.
        var warm = metadata.toBuilder();
        for (int i = 0; i < 100; i++) warm.putCacheKeys(keys[i], true);
        warm.build().toByteArray();
        new CacheFixtureWire.Frame(metadata, null,
                CacheFixtureWire.NaturalKeys.takeSnapshots(new long[]{7}, new long[0], null)).toProto();
        long before = allocations.getThreadAllocatedBytes(thread);
        var compact = new CacheFixtureWire.Frame(metadata, null,
                CacheFixtureWire.NaturalKeys.takeSnapshots(keys.clone(), new long[0], null));
        int compactBytes;
        try (var stream = compact.stream()) {
            compactBytes = ((Drainable)stream).drainTo(java.io.OutputStream.nullOutputStream());
        }
        long compactAllocated = allocations.getThreadAllocatedBytes(thread) - before;
        before = allocations.getThreadAllocatedBytes(thread);
        var builder = metadata.toBuilder();
        for (long key : keys) builder.putCacheKeys(key, true);
        var legacy = new CacheFixtureWire.Frame(builder.build(), null);
        int legacyBytes;
        try (var stream = legacy.stream()) {
            legacyBytes = ((Drainable)stream).drainTo(java.io.OutputStream.nullOutputStream());
        }
        long legacyAllocated = allocations.getThreadAllocatedBytes(thread) - before;
        assertEquals(legacyBytes, compactBytes);
        assertTrue(compactAllocated * 4 < legacyAllocated,
                "compact=" + compactAllocated + " legacy=" + legacyAllocated);
        System.out.println("CacheStatus allocation bytes: compact=" + compactAllocated + " legacy=" + legacyAllocated);
    }

    @Test void compactEncodingCliDefaultsOffAndRejectsInvalidBoolean() {
        var args = new java.util.ArrayList<>(java.util.List.of("--endpoint-file", "endpoints.json",
                "--engine-config", "../tools/mock_bundle/engine-config.json", "--calibration", "performance.json"));
        assertFalse(JavaMockEngineCluster.Config.parse(args.toArray(String[]::new)).cacheStatusCompactEncoding);
        args.addAll(java.util.List.of("--cache-status-compact-encoding", "true"));
        assertTrue(JavaMockEngineCluster.Config.parse(args.toArray(String[]::new)).cacheStatusCompactEncoding);
        args.set(args.size() - 1, "invalid");
        assertThrows(IllegalArgumentException.class,
                () -> JavaMockEngineCluster.Config.parse(args.toArray(String[]::new)));
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void unmodifiedGrpcClientDecodesFixtureAndOrdinaryResponses(boolean compact) throws Exception {
        var perf = directory.resolve("perf.json"); java.nio.file.Files.writeString(perf, "{\"block_size\":2}");
        var config = directory.resolve("master.json"); MockMasterConfig.writeWithPrefillExpression(config, "20");
        var model = MockModelFixture.load(perf.toString(), config.toString());
        try (var cluster = MockEngineTestCluster.create(model, 61580, 1, 0)) {
            var service = cluster.prefill(0);
            assertFalse(service.cacheStatusCompactEncoding, "extra Mock optimization defaults off");
            service.cacheStatusCompactEncoding = compact;
            var server = io.grpc.netty.NettyServerBuilder.forPort(0).addService(CacheFixtureWire.bind(service)).build().start();
            var channel = io.grpc.ManagedChannelBuilder.forAddress("127.0.0.1", server.getPort()).usePlaintext().build();
            try {
                var stub = org.flexlb.engine.grpc.RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3, java.util.concurrent.TimeUnit.SECONDS);
                var request = EngineRpcService.CacheVersionPB.newBuilder().setNeedCacheKeys(true).setLatestCacheVersion(-1).build();
                assertEquals(0, stub.getCacheStatus(request).getCacheKeysCount());
                assertTrue(stub.getWorkerStatus(EngineRpcService.StatusVersionPB.getDefaultInstance()).getAlive());
                service.seedNaturalCache(java.util.List.of(hash(0), hash(Long.MIN_VALUE), 7L));
                var natural = stub.getCacheStatus(request);
                assertEquals(3, natural.getCacheKeysCount());
                service.cacheFixture("configure", 0, 120, 176, 65, 0, 1000);
                var response = stub.getCacheStatus(request);
                var expected = reference(0, 120, 176, 65, 0, response.getVersion()).toBuilder()
                        .putCacheKeys(7L, true).build();
                assertTrue(response.getVersion() > natural.getVersion());
                assertEquals(expected.getCacheKeysMap(), response.getCacheKeysMap());
                CacheFixtureWire.Frame merged = MockEngineTestSupport.unary(o -> service.getCacheStatusWire(request, o));
                assertEquals(compact, merged.natural() != null);
                try (var stream = merged.stream()) {
                    assertEquals(response.getSerializedSize(), stream.available(), "overlapping keys occur only once on the wire");
                    assertEquals(response, EngineRpcService.CacheStatusPB.parseFrom(stream));
                }
                assertEquals(response, MockEngineTestSupport.<EngineRpcService.CacheStatusPB>unary(o -> service.getCacheStatus(request, o)));
                assertTrue(service.evictCacheKeys(java.util.List.of(7L)));
                assertEquals(response, merged.toProto(), "in-flight snapshot survives live cache eviction");
                var evicted = stub.getCacheStatus(request);
                assertEquals(241, evicted.getCacheKeysCount());
                assertEquals(evicted, stub.getCacheStatus(request.toBuilder().setLatestCacheVersion(evicted.getVersion()).build()),
                        "acknowledged versions still return full keys like real");
                assertTrue(evicted.getVersion() > response.getVersion());
                var scalar = stub.getCacheStatus(request.toBuilder().setNeedCacheKeys(false).build());
                assertEquals(evicted.getVersion(), scalar.getVersion());
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
