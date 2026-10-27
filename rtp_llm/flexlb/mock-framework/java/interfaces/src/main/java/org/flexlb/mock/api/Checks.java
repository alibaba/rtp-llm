package org.flexlb.mock.api;

import java.lang.annotation.*;

/** Cross-field constraints exported with the record's parameter schema. */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.TYPE)
public @interface Checks {
    Check[] value();
    @Retention(RetentionPolicy.RUNTIME)
    @Target({})
    @interface Check {
        Rule rule();
        String[] fields();
    }
    enum Rule { ordered }
}
