package org.flexlb.exception;

import lombok.Getter;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.enums.StatusEnum;

@Getter
public class WhaleException extends RuntimeException {

    private final int code;

    private final String name;

    /**
     * 是否是简单异常：仅落异常Msg，不落异常堆栈信息
     * @return boolean
     */
    public boolean isSimpleException() {
        return false;
    }

    /**
     * 是否是用户异常: 不落异常日志，直接返回给用户错误信息
     * @return boolean
     */
    public boolean isUserException() {
        return false;
    }

    public WhaleException(int code, String name, String message, Throwable cause) {
        super(message, cause);
        this.code = code;
        this.name = name;
    }

    public WhaleException(int code, String name, String message) {
        super(message);
        this.code = code;
        this.name = name;
    }

    public WhaleException(StatusEnum statusEnum) {
        super(statusEnum.getMessage());
        this.code = statusEnum.getCode();
        this.name = statusEnum.getName();
    }

    public WhaleException(BalanceStatusEnum balanceStatusEnum) {
        super(balanceStatusEnum.getMessage());
        this.code = balanceStatusEnum.getCode();
        this.name = balanceStatusEnum.name();
    }
}
