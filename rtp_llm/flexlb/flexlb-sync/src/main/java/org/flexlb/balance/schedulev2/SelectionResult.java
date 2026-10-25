package org.flexlb.balance.schedulev2;

/**
 * worker 准备的计算材料，或者准备时抛出的异常。由调度线程处理并关闭。
 */
public record SelectionResult<T extends AutoCloseable>(T value,
                                                       Throwable failure,
                                                       long preparedAtNanos)
        implements
            AutoCloseable {

    /**
     * 用单调时钟计时，调整系统时间不会延长材料有效期。
     */
    public boolean expired(long ttlMs) {
        return java.util.concurrent.TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - preparedAtNanos) >= ttlMs;
    }

    @Override
    public void close() throws Exception {
        if (value != null) {
            value.close();
        }
    }
}
