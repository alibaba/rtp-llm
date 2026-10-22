package org.flexlb.httpserver;

import org.flexlb.config.FlexlbConfig;

/** Startup policy for logical receiving wire-byte reservations, not physical Direct accounting. */
final class GrpcReceivingBudget {
    // Matches the server's 4 MiB receive window plus allocator/frame headroom. Wire callbacks
    // follow buffer retention and flow-credit return; this is a heuristic, never a Direct hard
    // bound.
    static final long INITIAL_RESERVATION_BYTES = 8L << 20;
    static final int MAX_RECEIVING_SLOTS = 1024;

    record Limits(
            int scheduleCalls,
            int controlCalls,
            long timeoutMs,
            long budgetBytes,
            long scheduleBudgetBytes,
            long controlBudgetBytes,
            long initialReservationBytes) {}

    static Limits resolve(FlexlbConfig.GrpcServerConfig config, long maxDirectBytes) {
        if (maxDirectBytes <= 0)
            throw new IllegalArgumentException(
                    "gRPC receiving budget requires a positive Direct limit");
        long budget =
                config.getReceiveBudgetBytes() == 0
                        ? maxDirectBytes / 4
                        : config.getReceiveBudgetBytes();
        if (budget <= 0
                || budget > maxDirectBytes
                || config.getReceiveTimeoutMs() <= 0
                || config.getMaxReceivingRequests() < 0
                || config.getMaxReceivingControlRequests() < 0) {
            throw new IllegalArgumentException("Invalid grpcServer receiving budget or timeout");
        }
        // At most half of each lane is reserved by HEADERS alone. Leave another base's worth
        // per slot for body charges, so a full HEADERS burst does not reject every tiny body.
        long slots = Math.min(MAX_RECEIVING_SLOTS, budget / (2 * INITIAL_RESERVATION_BYTES));
        if (slots < 2)
            throw new IllegalArgumentException(
                    "grpcServer.receiveBudgetBytes="
                            + budget
                            + " requires at least "
                            + (4 * INITIAL_RESERVATION_BYTES)
                            + " bytes for separate Schedule/Cancel lanes with body headroom");
        int controls =
                config.getMaxReceivingControlRequests() == 0
                        ? (int) Math.max(1, Math.min(8, slots / 8))
                        : config.getMaxReceivingControlRequests();
        int schedules =
                config.getMaxReceivingRequests() == 0
                        ? (int) slots - controls
                        : config.getMaxReceivingRequests();
        if (schedules < 1 || controls < 1 || (long) schedules + controls > slots) {
            throw new IllegalArgumentException(
                    "grpcServer receiving lane limits exceed reserved slots=" + slots);
        }
        long controlBytes = controls * (2 * INITIAL_RESERVATION_BYTES);
        long scheduleBytes = budget - controlBytes;
        return new Limits(
                schedules,
                controls,
                config.getReceiveTimeoutMs(),
                budget,
                scheduleBytes,
                controlBytes,
                INITIAL_RESERVATION_BYTES);
    }

    private GrpcReceivingBudget() {}
}
