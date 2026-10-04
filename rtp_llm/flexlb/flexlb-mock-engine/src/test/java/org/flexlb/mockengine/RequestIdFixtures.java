package org.flexlb.mockengine;

import com.google.protobuf.ByteString;
import com.google.protobuf.Message;
import com.google.protobuf.UnknownFieldSet;

/**
 * Writes Engine request IDs for this module's tests.
 * API, sync and mock-engine retain module-local fixtures because their test classes
 * are not shared dependencies. The copies use the same integer/string wire rules.
 */
public final class RequestIdFixtures {
    private RequestIdFixtures() {
    }

    public static <B extends Message.Builder> B write(B builder, String requestId) {
        var field = builder.getDescriptorForType().findFieldByName("request_id");
        builder.clearField(field);
        var unknown = builder.getUnknownFields().toBuilder().clearField(field.getNumber());
        try {
            long number = Long.parseLong(requestId);
            if (number != 0 && Long.toString(number).equals(requestId)) {
                builder.setField(field, number);
                builder.setUnknownFields(unknown.build());
                return builder;
            }
        } catch (NumberFormatException ignored) {
        }
        unknown.addField(field.getNumber(), UnknownFieldSet.Field.newBuilder()
                .addLengthDelimited(ByteString.copyFromUtf8(requestId)).build());
        builder.setUnknownFields(unknown.build());
        return builder;
    }
}
