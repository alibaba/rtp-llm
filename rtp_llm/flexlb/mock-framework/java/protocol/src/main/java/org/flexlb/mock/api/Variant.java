package org.flexlb.mock.api;

import java.lang.annotation.*;

/** Wire representation of a sealed parameter variant. */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.TYPE)
public @interface Variant {
    String value() default "";
    boolean scalar() default false;
}
