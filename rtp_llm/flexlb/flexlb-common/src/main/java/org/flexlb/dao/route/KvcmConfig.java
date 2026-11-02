package org.flexlb.dao.route;

import lombok.Data;

/** MetaService bootstrap address and shared cache namespace, as in MU. */
@Data
public final class KvcmConfig {
    private String address;
    private int port = 6381;
    private String namespace;
}
