package org.flexlb.util.network;

/**
 * Resolves and validates the message-size limit shared by FlexLB gRPC endpoints.
 */
public final class GrpcConstants {
    // Set the same MiB limit on frontend and FlexLB; read once at process startup.
    public static final int MAX_MESSAGE_SIZE =
            parseMaxMessageSize(System.getenv("FLEXLB_MAX_MESSAGE_SIZE_MB"));

    private GrpcConstants() {}

    static int parseMaxMessageSize(String value) {
        String configured = value == null || value.trim().isEmpty() ? "512" : value.trim();
        try {
            int mib = Integer.parseInt(configured);
            if (!configured.matches("[0-9]+") || mib < 1 || mib > 2047) {
                throw new NumberFormatException();
            }
            return mib * 1024 * 1024;
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "FLEXLB_MAX_MESSAGE_SIZE_MB must be an integer from 1 to 2047 (MiB)", e);
        }
    }
}
