package org.flexlb.balance.queue;

/**
 * 入队和交付在出错后都要继续做完回滚或通知，再把第一个异常抛出去，后面的异常挂在它下面。这里放两边共用的处理方法。
 */
public final class QueueFailures {

    private QueueFailures() {
    }

    /**
     * 把后一个异常挂到第一个异常下面并返回第一个异常；第一个为 null 时返回后一个。
     */
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

    /**
     * 把收集到的异常转成要抛出的异常：RuntimeException 原样返回，Error 直接抛出，
     * 其他异常包成 IllegalStateException。
     */
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
