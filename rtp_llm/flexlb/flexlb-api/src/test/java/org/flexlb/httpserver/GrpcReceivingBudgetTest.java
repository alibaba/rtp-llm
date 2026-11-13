package org.flexlb.httpserver;

import org.flexlb.config.FlexlbConfig;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GrpcReceivingBudgetTest {
  @Test
  void defaultCountsDoNotDependOnDirectMemory() {
    for (long direct : new long[] {2L << 30, 8L << 30, 16L << 30, 64L << 30}) {
      var b = GrpcReceivingBudget.resolve(new FlexlbConfig.GrpcServerConfig(), direct);
      assertEquals(1024, b.scheduleCalls());
      assertEquals(128, b.controlCalls());
      assertEquals(direct / 4, b.budgetBytes());
      assertEquals(30000, b.timeoutMs());
      assertEquals(b.budgetBytes(), b.scheduleBudgetBytes() + b.controlBudgetBytes());
    }
  }

  @Test
  void zeroCountsRemainAnAliasForFixedDefaults() {
    var c = new FlexlbConfig.GrpcServerConfig();
    c.setMaxReceivingRequests(0);
    c.setMaxReceivingControlRequests(0);
    var b = GrpcReceivingBudget.resolve(c, 2L << 30);
    assertEquals(1024, b.scheduleCalls());
    assertEquals(128, b.controlCalls());
  }

  @Test
  void configuredCountsAreIndependentOfBodyBudget() {
    var c = new FlexlbConfig.GrpcServerConfig();
    c.setMaxReceivingRequests(4096);
    c.setMaxReceivingControlRequests(32);
    c.setReceiveBudgetBytes(64L << 20);
    var b = GrpcReceivingBudget.resolve(c, 2L << 30);
    assertEquals(4096, b.scheduleCalls());
    assertEquals(32, b.controlCalls());
    assertEquals(56L << 20, b.scheduleBudgetBytes());
    assertEquals(8L << 20, b.controlBudgetBytes());
    c.setMaxReceivingRequests(1);
    c.setMaxReceivingControlRequests(1);
    var reduced = GrpcReceivingBudget.resolve(c, 2L << 30);
    assertEquals(b.scheduleBudgetBytes(), reduced.scheduleBudgetBytes());
    assertEquals(b.controlBudgetBytes(), reduced.controlBudgetBytes());
  }

  @Test
  void controlBudgetIsReservedIndependentlyAndCapped() {
    var c = new FlexlbConfig.GrpcServerConfig();
    var smaller = GrpcReceivingBudget.resolve(c, 2L << 30);
    assertEquals(64L << 20, smaller.controlBudgetBytes());
    assertEquals(448L << 20, smaller.scheduleBudgetBytes());
    var larger = GrpcReceivingBudget.resolve(c, 16L << 30);
    assertEquals(128L << 20, larger.controlBudgetBytes());
    assertEquals(3968L << 20, larger.scheduleBudgetBytes());
  }

  @Test
  void bodyBudgetMustFitDirectAndLeaveBothLanesPositive() {
    var c = new FlexlbConfig.GrpcServerConfig();
    c.setReceiveBudgetBytes(1L << 20);
    var b = GrpcReceivingBudget.resolve(c, 8L << 30);
    assertEquals(1024, b.scheduleCalls());
    assertTrue(b.controlBudgetBytes() > 0);
    assertTrue(b.scheduleBudgetBytes() > 0);
    c.setReceiveBudgetBytes(9L << 30);
    assertThrows(IllegalArgumentException.class, () -> GrpcReceivingBudget.resolve(c, 8L << 30));
    c.setReceiveBudgetBytes(-1);
    assertThrows(IllegalArgumentException.class, () -> GrpcReceivingBudget.resolve(c, 8L << 30));
    c.setReceiveBudgetBytes(7);
    assertThrows(IllegalArgumentException.class, () -> GrpcReceivingBudget.resolve(c, 8L << 30));
    assertThrows(IllegalArgumentException.class, () -> GrpcReceivingBudget.resolve(c, 0));
  }

  @Test
  void rejectsNegativeCountsAndNonPositiveTimeout() {
    var c = new FlexlbConfig.GrpcServerConfig();
    c.setMaxReceivingRequests(-1);
    assertThrows(IllegalArgumentException.class, () -> GrpcReceivingBudget.resolve(c, 64L << 30));
    c.setMaxReceivingRequests(1024);
    c.setMaxReceivingControlRequests(-1);
    assertThrows(IllegalArgumentException.class, () -> GrpcReceivingBudget.resolve(c, 64L << 30));
    c.setMaxReceivingControlRequests(8);
    c.setReceiveTimeoutMs(0);
    assertThrows(IllegalArgumentException.class, () -> GrpcReceivingBudget.resolve(c, 64L << 30));
  }
}
