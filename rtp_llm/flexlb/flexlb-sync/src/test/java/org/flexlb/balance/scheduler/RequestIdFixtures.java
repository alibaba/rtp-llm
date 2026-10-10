package org.flexlb.balance.scheduler;

import com.google.protobuf.Message;

/**
 * Writes canonical int64 Engine request IDs for this module's tests.
 */
public final class RequestIdFixtures {
    private RequestIdFixtures() {
    }

    public static <B extends Message.Builder> B write(B builder, String requestId) {
        long number;
        try {
            number = Long.parseLong(requestId);
        } catch (NumberFormatException error) {
            throw new IllegalArgumentException("Engine request ID must be a canonical int64: " + requestId, error);
        }
        if (!Long.toString(number).equals(requestId)) {
            throw new IllegalArgumentException("Engine request ID must be a canonical int64: " + requestId);
        }
        var field = builder.getDescriptorForType().findFieldByName("request_id");
        builder.clearField(field);
        builder.setUnknownFields(builder.getUnknownFields().toBuilder().clearField(field.getNumber()).build());
        builder.setField(field, number);
        return builder;
    }
}
