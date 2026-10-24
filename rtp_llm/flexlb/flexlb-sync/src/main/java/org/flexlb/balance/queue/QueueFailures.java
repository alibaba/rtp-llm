package org.flexlb.balance.queue;

/**
 * 入队和交付在出错后都要继续做完回滚或通知，再把第一个异常抛出去，后面的异常挂在它下面。这里放两边共用的处理方法。
 */
public final class QueueFailures {

    private QueueFailures() {
    }

    public static Throwable appendFailure(
            Throwable first,
            Throwable next) {
        if (first == null) {
            return next;
        }
        if (first != next) {
            first.addSuppressed(next);
        }
        return first;
    }

    /** Release the publish-time reservation only while it is still item-owned. */
    public static RuntimeException propagateCommitFailure(
            Throwable failure) {
        if (failure instanceof RuntimeException runtimeFailure) {
            return runtimeFailure;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        return new IllegalStateException(
                "delivery selection failed after ownership commit", failure);
    }
}
