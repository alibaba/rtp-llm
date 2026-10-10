package org.flexlb.balance.prediction;

import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import static org.flexlb.balance.prediction.ArithmeticFormulaAst.*;

/** Recursive-descent grammar and binding/aggregate validation. */
final class ArithmeticFormulaParser {
    private final String input;
    private final Map<String, ConstantNode> parameters = new LinkedHashMap<>();
    private final Map<String, Integer> variables;
    private final Set<String> aggregateExcludedVariables;
    private final Set<String> referencedVariables = new HashSet<>();
    private final boolean allowAggregates;
    private final int variableCount;
    private int pos;
    private int aggregateDepth;

    ArithmeticFormulaParser(String input, Map<String, Integer> variables,
           Set<String> aggregateExcludedVariables, boolean allowAggregates) {
        this.input = input;
        this.variables = Map.copyOf(Objects.requireNonNull(variables, "variables"));
        this.aggregateExcludedVariables = Set.copyOf(
                Objects.requireNonNull(aggregateExcludedVariables, "aggregateExcludedVariables"));
        this.allowAggregates = allowAggregates;
        int count = 0;
        for (int index : this.variables.values()) {
            if (index < 0 || index == Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Variable index is outside its domain: " + index);
            }
            count = Math.max(count, index + 1);
        }
        this.variableCount = count;
    }

    Node parse() {
        Node result = parseExpression();
        expectEnd();
        return result;
    }

    Set<String> referencedVariables() {
        return Set.copyOf(referencedVariables);
    }

    // expression → term (('+' | '-') term)*
    private Node parseExpression() {
        Node node = parseTerm();
        while (true) {
            skipWs();
            if (match('+')) {
                node = binary('+', node, parseTerm());
            } else if (match('-')) {
                node = binary('-', node, parseTerm());
            } else {
                return node;
            }
        }
    }

    // term → factor (('*' | '/') factor)*
    private Node parseTerm() {
        Node node = parseFactor();
        while (true) {
            skipWs();
            if (match('*')) {
                node = binary('*', node, parseFactor());
            } else if (match('/')) {
                node = binary('/', node, parseFactor());
            } else {
                return node;
            }
        }
    }

    // factor → unary ('^' factor)*    right-associative
    private Node parseFactor() {
        Node node = parseUnary();
        skipWs();
        return match('^') ? binary('^', node, parseFactor()) : node;
    }

    // unary → ('+' | '-') unary | primary
    private Node parseUnary() {
        skipWs();
        if (match('+')) {
            return unary('+', parseUnary());
        }
        if (match('-')) {
            return unary('-', parseUnary());
        }
        return parsePrimary();
    }

    // primary → '(' expression ')' | function_call | param_call | number | variable
    private Node parsePrimary() {
        skipWs();
        if (match('(')) {
            Node node = parseExpression();
            skipWs();
            if (!match(')')) {
                throw error("Expected ')'");
            }
            return node;
        }
        if (hasNext() && Character.isLetter(peek())) {
            String name = parseIdentifier();
            skipWs();
            if (match('(')) {
                if (name.equals("param")) {
                    return parseParamCall();
                }
                return parseFuncCall(name);
            }
            if (name.equals("param")) {
                throw error("'param' must be used as param(name, initialValue)");
            }
            if (aggregateDepth > 0 && aggregateExcludedVariables.contains(name)) {
                throw error("Batch-scoped variable cannot be used inside sum(): " + name);
            }
            Integer idx = variables.get(name);
            if (idx == null) {
                throw error("Unknown variable: " + name);
            }
            referencedVariables.add(name);
            return new VariableNode(idx);
        }
        if (hasNext() && (Character.isDigit(peek()) || peek() == '.')) {
            return parseNumber();
        }
        throw error("Expected number, variable, or '('");
    }

    // Parameters become constants while their initializer variable references are retained.
    private Node parseParamCall() {
        skipWs();
        if (!hasNext() || !(Character.isLetter(peek()) || peek() == '_')) {
            throw error("Expected parameter name in param()");
        }
        String paramName = parseIdentifier();
        skipWs();
        if (!match(',')) {
            throw error("Expected ',' after parameter name in param()");
        }
        skipWs();
        Node initialValueNode = parseExpression();
        double initialValue = evaluate(initialValueNode, new double[variableCount], null);
        skipWs();
        if (!match(')')) {
            throw error("Expected ')' after param() arguments");
        }
        ConstantNode existing = parameters.get(paramName);
        if (existing != null) {
            if (existing.value() != initialValue) {
                throw error("Inconsistent initial value for parameter '" + paramName
                        + "': " + existing.value() + " vs " + initialValue);
            }
            return existing;
        }
        ConstantNode node = new ConstantNode(initialValue);
        parameters.put(paramName, node);
        return node;
    }

    private Node parseFuncCall(String name) {
        Function function = Function.named(name);
        boolean aggregate = name.equals("sum");
        if (function == null && !(allowAggregates && aggregate)) {
            throw error("Unknown function: " + name);
        }
        skipWs();
        if (aggregate) {
            aggregateDepth++;
        }
        Node arg0;
        try {
            arg0 = parseExpression();
        } finally {
            if (aggregate) {
                aggregateDepth--;
            }
        }
        if (aggregate) {
            skipWs();
            if (!match(')')) {
                throw error("Expected ')' after aggregate function argument");
            }
            return new AggregateFuncNode(arg0);
        }
        if (function.arity == 2) {
            skipWs();
            if (!match(',')) {
                throw error("Expected ',' in binary function '" + name + "'");
            }
            skipWs();
            Node arg1 = parseExpression();
            skipWs();
            if (!match(')')) {
                throw error("Expected ')' after function arguments");
            }
            return function(function, arg0, arg1);
        }
        skipWs();
        if (!match(')')) {
            throw error("Expected ')' after function argument");
        }
        return function(function, arg0, null);
    }

    private Node parseNumber() {
        int start = pos;
        while (hasNext() && (Character.isDigit(peek()) || peek() == '.')) {
            pos++;
        }
        if (hasNext() && (peek() == 'e' || peek() == 'E')) {
            pos++;
            if (hasNext() && (peek() == '+' || peek() == '-')) {
                pos++;
            }
            while (hasNext() && Character.isDigit(peek())) {
                pos++;
            }
        }
        try {
            return new ConstantNode(Double.parseDouble(input.substring(start, pos)));
        } catch (NumberFormatException e) {
            throw error("Invalid number");
        }
    }

    private String parseIdentifier() {
        int start = pos;
        while (hasNext() && (Character.isLetterOrDigit(peek()) || peek() == '_')) {
            pos++;
        }
        return input.substring(start, pos);
    }

    private void expectEnd() {
        skipWs();
        if (hasNext()) {
            throw error("Unexpected token");
        }
    }

    // ---- helpers ----

    private boolean match(char expected) {
        if (hasNext() && peek() == expected) {
            pos++;
            return true;
        }
        return false;
    }

    private char peek() {
        return input.charAt(pos);
    }

    private boolean hasNext() {
        return pos < input.length();
    }

    private void skipWs() {
        while (hasNext() && Character.isWhitespace(peek())) {
            pos++;
        }
    }

    private IllegalArgumentException error(String msg) {
        return new IllegalArgumentException(
                msg + " at pos " + pos + " in: " + input);
    }
}
