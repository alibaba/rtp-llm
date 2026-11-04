package org.flexlb.mock.api;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.lang.annotation.ElementType;

/** Component method identity; Python owns step execution and expectations. */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface Endpoint {
    String value();
    Kind kind();
    boolean background() default false;
    String unit() default "";
    enum Kind { action, read }
}
