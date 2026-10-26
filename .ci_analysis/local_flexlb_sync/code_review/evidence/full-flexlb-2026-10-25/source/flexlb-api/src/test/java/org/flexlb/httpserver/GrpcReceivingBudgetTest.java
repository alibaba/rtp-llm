package org.flexlb.httpserver;

import org.flexlb.config.FlexlbConfig;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GrpcReceivingBudgetTest {
    @Test
    void default64GiBHasBodyHeadroomAndBoundedCounts() {
        var b = GrpcReceivingBudget.resolve(new FlexlbConfig.GrpcServerConfig(), 64L << 30);
        assertEquals(1016, b.scheduleCalls());
        assertEquals(8, b.controlCalls());
        assertEquals(16L << 30, b.budgetBytes());
        assertEquals(128L << 20, b.controlBudgetBytes());
        assertEquals(16256L << 20, b.scheduleBudgetBytes());
        assertEquals(8L << 20, b.initialReservationBytes());
        assertEquals(30000, b.timeoutMs());
        headroom(b);
    }

    @Test
    void default8GiBKeepsNormalSmallMessageCapacity() {
        var b = GrpcReceivingBudget.resolve(new FlexlbConfig.GrpcServerConfig(), 8L << 30);
        assertEquals(120, b.scheduleCalls());
        assertEquals(8, b.controlCalls());
        assertEquals(2L << 30, b.budgetBytes());
        assertEquals(1920L << 20, b.scheduleBudgetBytes());
        headroom(b);
    }

    @Test
    void smallerDirectUsesSamePolicyWithoutWorstCaseMessageReservation() {
        var c = new FlexlbConfig.GrpcServerConfig();
        var b = GrpcReceivingBudget.resolve(c, 2L << 30);
        assertEquals(28, b.scheduleCalls());
        assertEquals(4, b.controlCalls());
        assertEquals(512L << 20, b.budgetBytes());
        headroom(b);
        assertThrows(
                IllegalArgumentException.class, () -> GrpcReceivingBudget.resolve(c, 64L << 20));
    }

    @Test
    void explicitBudgetMustFitDirectAndBothLanes() {
        var c = new FlexlbConfig.GrpcServerConfig();
        c.setReceiveBudgetBytes(31L << 20);
        assertThrows(
                IllegalArgumentException.class, () -> GrpcReceivingBudget.resolve(c, 8L << 30));
        c.setReceiveBudgetBytes(32L << 20);
        var b = GrpcReceivingBudget.resolve(c, 8L << 30);
        assertEquals(1, b.scheduleCalls());
        assertEquals(1, b.controlCalls());
        headroom(b);
        c.setReceiveBudgetBytes(9L << 30);
        assertThrows(
                IllegalArgumentException.class, () -> GrpcReceivingBudget.resolve(c, 8L << 30));
        c.setReceiveBudgetBytes(-1);
        assertThrows(
                IllegalArgumentException.class, () -> GrpcReceivingBudget.resolve(c, 8L << 30));
    }

    @Test
    void explicitCountsCannotConsumeBodyHeadroomOrExceedTotalCap() {
        var c = new FlexlbConfig.GrpcServerConfig();
        c.setMaxReceivingRequests(1024);
        assertThrows(
                IllegalArgumentException.class, () -> GrpcReceivingBudget.resolve(c, 64L << 30));
        c.setMaxReceivingRequests(32);
        c.setMaxReceivingControlRequests(2);
        var b = GrpcReceivingBudget.resolve(c, 64L << 30);
        assertEquals(32, b.scheduleCalls());
        assertEquals(2, b.controlCalls());
        assertEquals(32L << 20, b.controlBudgetBytes());
        headroom(b);
        c.setMaxReceivingControlRequests(1024);
        assertThrows(
                IllegalArgumentException.class, () -> GrpcReceivingBudget.resolve(c, 64L << 30));
    }

    @Test
    void hugeDirectStillCapsConcurrentReceivingAt1024() {
        var b = GrpcReceivingBudget.resolve(new FlexlbConfig.GrpcServerConfig(), 1L << 40);
        assertEquals(1024, b.scheduleCalls() + b.controlCalls());
        headroom(b);
    }

    static void headroom(GrpcReceivingBudget.Limits b) {
        assertTrue(b.scheduleCalls() * b.initialReservationBytes() <= b.scheduleBudgetBytes() / 2);
        assertTrue(b.controlCalls() * b.initialReservationBytes() <= b.controlBudgetBytes() / 2);
        assertEquals(b.budgetBytes(), b.scheduleBudgetBytes() + b.controlBudgetBytes());
    }
}
