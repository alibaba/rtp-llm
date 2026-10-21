package org.flexlb.engine.grpc;

import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.FlexlbScheduleRequestPB;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class SessionAffinityProtocolTest {

    @Test
    void preservesFieldNumbersAndNegativeBucketAcrossSerialization() throws Exception {
        var descriptor = FlexlbScheduleRequestPB.getDescriptor();
        assertEquals(22, descriptor.findFieldByName("session_affinity_id").getNumber());
        assertEquals(23, descriptor.findFieldByName("session_affinity_bucket_id").getNumber());
        assertEquals(24, descriptor.findFieldByName("session_affinity_type").getNumber());
        var request = FlexlbScheduleRequestPB.newBuilder()
                .setSessionAffinityId("session")
                .setSessionAffinityBucketId(-1)
                .setSessionAffinityType("user")
                .build();
        var forwarded = FlexlbScheduleRequestPB.parseFrom(request.toByteArray()).toBuilder()
                .setForwardHop(1).build();
        assertEquals("session", forwarded.getSessionAffinityId());
        assertEquals(-1, forwarded.getSessionAffinityBucketId());
        assertEquals("user", forwarded.getSessionAffinityType());
    }

    @Test
    void scalarDefaultsRetainZeroBucketForOldClients() {
        var request = FlexlbScheduleRequestPB.getDefaultInstance();
        assertEquals("", request.getSessionAffinityId());
        assertEquals(0, request.getSessionAffinityBucketId());
        assertEquals("", request.getSessionAffinityType());
    }
}
