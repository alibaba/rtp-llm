package org.flexlb.balance.dp;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PrefillProfilerTest {

    @Test
    void fitPolynomial_constant_function() {
        // T(n) = 190 for all n
        List<double[]> points = List.of(
                new double[]{32, 190},
                new double[]{128, 190},
                new double[]{512, 190},
                new double[]{1024, 190},
                new double[]{2048, 190}
        );
        double[] c = PrefillProfiler.fitPolynomial(points);
        assertEquals(190, c[0], 1.0);
        assertEquals(0, c[1], 0.001);
        assertEquals(0, c[2], 1e-6);
    }

    @Test
    void fitPolynomial_linear_function() {
        // T(n) = 50 + 0.2*n
        List<double[]> points = new ArrayList<>();
        for (int n : new int[]{32, 64, 128, 256, 512, 1024, 2048}) {
            points.add(new double[]{n, 50 + 0.2 * n});
        }
        double[] c = PrefillProfiler.fitPolynomial(points);
        assertEquals(50, c[0], 1.0);
        assertEquals(0.2, c[1], 0.001);
        assertEquals(0, c[2], 1e-6);
    }

    @Test
    void fitPolynomial_quadratic_function() {
        // T(n) = 100 + 0.05*n + 0.0001*n²
        List<double[]> points = new ArrayList<>();
        for (int n : new int[]{32, 64, 128, 256, 512, 1024, 2048}) {
            points.add(new double[]{n, 100 + 0.05 * n + 0.0001 * n * n});
        }
        double[] c = PrefillProfiler.fitPolynomial(points);
        assertEquals(100, c[0], 1.0);
        assertEquals(0.05, c[1], 0.01);
        assertEquals(0.0001, c[2], 1e-5);
    }

    @Test
    void fitPolynomial_noisy_constant() {
        // T(n) ≈ 190 with noise
        List<double[]> points = List.of(
                new double[]{32, 185},
                new double[]{64, 192},
                new double[]{128, 188},
                new double[]{256, 195},
                new double[]{512, 187},
                new double[]{1024, 191},
                new double[]{2048, 193}
        );
        double[] c = PrefillProfiler.fitPolynomial(points);
        // c0 should be close to 190
        assertTrue(c[0] > 170 && c[0] < 210, "c0=" + c[0]);
        // c1 and c2 should be near zero
        assertTrue(Math.abs(c[1]) < 0.05, "c1=" + c[1]);
        assertTrue(Math.abs(c[2]) < 1e-4, "c2=" + c[2]);
    }

    @Test
    void fitPolynomial_minimum_three_points() {
        List<double[]> points = List.of(
                new double[]{100, 200},
                new double[]{500, 200},
                new double[]{1000, 200}
        );
        double[] c = PrefillProfiler.fitPolynomial(points);
        assertEquals(3, c.length);
        assertEquals(200, c[0], 1.0);
    }

    @Test
    void parseTokenLengths_default() {
        int[] lengths = PrefillProfiler.parseTokenLengths(null);
        assertArrayEquals(new int[]{32, 64, 128, 256, 512, 1024, 2048}, lengths);
    }

    @Test
    void parseTokenLengths_custom() {
        int[] lengths = PrefillProfiler.parseTokenLengths("16, 64, 512");
        assertArrayEquals(new int[]{16, 64, 512}, lengths);
    }

    @Test
    void parseTokenLengths_blank_uses_default() {
        int[] lengths = PrefillProfiler.parseTokenLengths("  ");
        assertArrayEquals(new int[]{32, 64, 128, 256, 512, 1024, 2048}, lengths);
    }
}
