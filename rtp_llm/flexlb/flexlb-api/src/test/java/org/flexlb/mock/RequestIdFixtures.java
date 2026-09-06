package org.flexlb.mock;

import com.google.protobuf.ByteString;
import com.google.protobuf.Message;
import com.google.protobuf.UnknownFieldSet;

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
