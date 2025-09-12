package org.flexlb.dao.route;

import lombok.Getter;

@Getter
public enum RoleType {
    PDFUSION("RoleType.PDFUSION", "PD融合"),
    PREFILL("RoleType.PREFILL", "预填充"),
    DECODE("RoleType.DECODE", "解码"),
    VIT("RoleType.VIT", "Vision Transformer");

    private final String code;
    private final String description;

    RoleType(String code, String description) {
        this.code = code;
        this.description = description;
    }

    /**
     * 检查字符串是否匹配当前角色类型
     */
    public boolean matches(String code) {
        return this.code.equals(code);
    }
}
