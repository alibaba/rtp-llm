package org.flexlb.exception;

import lombok.Getter;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.enums.StatusEnum;

public class FlexLBException extends RuntimeException {

    @Getter
    private final int code;

    @Getter
    private final String name;

    public FlexLBException(int code, String name, String message, Throwable cause) {
        super(message, cause);
        this.code = code;
        this.name = name;
    }

    public FlexLBException(int code, String name, String message) {
        super(message);
        this.code = code;
        this.name = name;
    }

    public FlexLBException(StatusEnum statusEnum) {
        super(statusEnum.getMessage());
        this.code = statusEnum.getCode();
        this.name = statusEnum.getName();
    }

    public FlexLBException(BalanceStatusEnum balanceStatusEnum) {
        super(balanceStatusEnum.getMessage());
        this.code = balanceStatusEnum.getCode();
        this.name = balanceStatusEnum.name();
    }
}
