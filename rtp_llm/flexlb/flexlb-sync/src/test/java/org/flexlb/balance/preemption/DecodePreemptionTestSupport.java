package org.flexlb.balance.preemption;

/**
 * Creates mocks for the public DecodePreemption entry point.
 */
public final class DecodePreemptionTestSupport {
    private DecodePreemptionTestSupport() {}

    public static DecodePreemption mock() {
        return org.mockito.Mockito.mock(DecodePreemption.class);
    }
}
