package org.flexlb.balance.endpoint.prefill;

/**
 * 申请 Prefill 预占的结果：成功时带着预占对象，失败时只带失败原因。
 */
public record ReservationResult<R extends AutoCloseable>(Status status, R reservation) {

    /**
     * 预占成功，或者没成功的原因。
     */
    public enum Status {
        SUCCESS,
        CAPACITY_FULL,
        /**
         * 请求已经不在这台 Prefill 的等待队列里：被取消、过期后移走，被同 id 的新请求替换，或者队列正在停止。
         */
        REQUEST_LEFT_QUEUE,
        REQUEST_ALREADY_RESERVED,
        BATCH_ID_ALREADY_RESERVED,
        ENDPOINT_RETIRED
    }

    public ReservationResult {
        if ((status == Status.SUCCESS) != (reservation != null)) {
            throw new IllegalArgumentException("only SUCCESS may carry a reservation");
        }
    }

    /**
     * 预占成功，带着拿到的预占对象。
     */
    public static <R extends AutoCloseable> ReservationResult<R> success(R reservation) {
        return new ReservationResult<>(Status.SUCCESS, reservation);
    }

    /**
     * 预占没成功，只带失败原因。
     */
    public static <R extends AutoCloseable> ReservationResult<R> fail(Status status) {
        return new ReservationResult<>(status, null);
    }

    /**
     * 是否拿到了预占。
     */
    public boolean isSuccess() {
        return status == Status.SUCCESS;
    }
}
