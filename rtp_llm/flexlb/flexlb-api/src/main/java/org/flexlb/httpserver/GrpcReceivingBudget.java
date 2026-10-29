package org.flexlb.httpserver;

import org.flexlb.config.FlexlbConfig;

/** Independent receiving count limits and logical body-byte budgets, not physical Direct accounting. */
final class GrpcReceivingBudget {
  private static final long MAX_CONTROL_BUDGET_BYTES = 128L << 20;

  record Limits(
      int scheduleCalls,
      int controlCalls,
      long timeoutMs,
      long budgetBytes,
      long scheduleBudgetBytes,
      long controlBudgetBytes) {}

  static Limits resolve(FlexlbConfig.GrpcServerConfig config, long maxDirectBytes) {
    if (maxDirectBytes <= 0)
      throw new IllegalArgumentException("gRPC receiving budget requires a positive Direct limit");
    long budget =
        config.getReceiveBudgetBytes() == 0 ? maxDirectBytes / 4 : config.getReceiveBudgetBytes();
    if (budget < 8
        || budget > maxDirectBytes
        || config.getReceiveTimeoutMs() <= 0
        || config.getMaxReceivingRequests() < 0
        || config.getMaxReceivingControlRequests() < 0) {
      throw new IllegalArgumentException("Invalid grpcServer receiving budget or timeout");
    }
    int controls =
        config.getMaxReceivingControlRequests() == 0
            ? FlexlbConfig.GrpcServerConfig.DEFAULT_MAX_RECEIVING_CONTROL_REQUESTS
            : config.getMaxReceivingControlRequests();
    int schedules =
        config.getMaxReceivingRequests() == 0
            ? FlexlbConfig.GrpcServerConfig.DEFAULT_MAX_RECEIVING_REQUESTS
            : config.getMaxReceivingRequests();
    // Keep Cancel bytes independent of Schedule pressure, without pricing each RPC at a fixed size.
    long controlBytes = Math.min(MAX_CONTROL_BUDGET_BYTES, budget / 8);
    long scheduleBytes = budget - controlBytes;
    return new Limits(
        schedules,
        controls,
        config.getReceiveTimeoutMs(),
        budget,
        scheduleBytes,
        controlBytes);
  }

  private GrpcReceivingBudget() {}
}
