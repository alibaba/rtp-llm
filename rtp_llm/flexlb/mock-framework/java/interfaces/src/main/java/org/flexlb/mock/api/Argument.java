package org.flexlb.mock.api;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.lang.annotation.ElementType;

/** Constraints used to generate the compiler's interface catalog. */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.RECORD_COMPONENT)
public @interface Argument {
    boolean optional() default false;
    double minimum() default Double.NaN;
    String[] ranges() default {};
    int minItems() default 0;
    double itemMinimum() default Double.NaN;
}
