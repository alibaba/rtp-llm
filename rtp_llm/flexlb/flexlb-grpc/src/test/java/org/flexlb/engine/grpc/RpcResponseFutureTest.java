package org.flexlb.engine.grpc;

import io.grpc.KnownLength;
import io.grpc.protobuf.ProtoUtils;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;
import java.io.ByteArrayInputStream;
import java.util.concurrent.CompletableFuture;
import static org.junit.jupiter.api.Assertions.*;

class RpcResponseFutureTest {
    private static final class KnownStream extends ByteArrayInputStream implements KnownLength {
        KnownStream(byte[] bytes) { super(bytes); }
    }
    @Test void largeMapBytesComeFromDeframerWithoutRecomputingProtobufSize() {
        var builder = EngineRpcService.CacheStatusPB.newBuilder().setVersion(1);
        for(long i=0;i<10000;i++) builder.putCacheKeys(i,true);
        byte[] wire=builder.build().toByteArray();
        var future=new RpcResponseFuture<EngineRpcService.CacheStatusPB>();
        var decoded=future.measuring(ProtoUtils.marshaller(EngineRpcService.CacheStatusPB.getDefaultInstance())).parse(new KnownStream(wire));
        assertEquals(wire.length,future.responseBytes());
        assertEquals(10000,decoded.getCacheKeysCount());
        // Generated size() would allocate a MapEntry per key; measurement must not invoke it.
        assertEquals(-1,ReflectionTestUtils.getField(decoded,"memoizedSize"));
        future.complete(decoded);
        assertEquals(wire.length,future.thenApply(ignored -> RpcResponseFuture.responseBytes(future)).join());
        assertEquals(-1,ReflectionTestUtils.getField(decoded,"memoizedSize"));
    }
    @Test void unknownLengthStreamCountsReadsAndMissingMetadataStaysUnknown() {
        byte[] wire=EngineRpcService.WorkerStatusPB.newBuilder().setAlive(true).setStatusVersion(1).build().toByteArray();
        var future=new RpcResponseFuture<EngineRpcService.WorkerStatusPB>();
        var decoded=future.measuring(ProtoUtils.marshaller(EngineRpcService.WorkerStatusPB.getDefaultInstance())).parse(new ByteArrayInputStream(wire));
        assertTrue(decoded.getAlive());assertEquals(wire.length,future.responseBytes());
        assertEquals(-1,RpcResponseFuture.responseBytes(CompletableFuture.completedFuture(decoded)));
    }
}
