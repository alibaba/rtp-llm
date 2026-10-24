package org.flexlb.mockengine;

/** Allocator materialization outcome, distinct from an ordinary copy failure. */
final class MockLoadFailure extends RuntimeException {
    enum Kind { RETRYABLE, PERMANENT, INTERNAL }
    final Kind kind;
    MockLoadFailure(Kind kind) { super("allocator load materialization " + kind); this.kind = kind; }
    static MockLoadFailure from(Throwable error) {
        while (error != null) {
            if (error instanceof MockLoadFailure failure) return failure;
            error = error.getCause();
        }
        return null;
    }
}
