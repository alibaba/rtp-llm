package org.flexlb.dao.master;

final class PrefillTimeFormula {
    private final String formula;
    private final Node root;

    private PrefillTimeFormula(String formula, Node root) {
        this.formula = formula;
        this.root = root;
    }

    static PrefillTimeFormula parse(String formula) {
        Parser parser = new Parser(formula);
        Node root = parser.parseExpression();
        parser.expectEnd();
        return new PrefillTimeFormula(formula, root);
    }

    long estimate(long tokens, long hitCacheTokens) {
        return (long) root.evaluate(tokens, hitCacheTokens);
    }

    @Override
    public String toString() {
        return formula;
    }

    private interface Node {
        double evaluate(long tokens, long hitCacheTokens);
    }

    private record ConstantNode(double value) implements Node {
        @Override
        public double evaluate(long tokens, long hitCacheTokens) {
            return value;
        }
    }

    private record VariableNode(String name) implements Node {
        @Override
        public double evaluate(long tokens, long hitCacheTokens) {
            return switch (name) {
                case "tokens", "inputLength", "input_length" -> tokens;
                case "hitCacheTokens", "prefixLength", "hit_cache_tokens", "prefix_length" -> hitCacheTokens;
                default -> throw new IllegalArgumentException("Unsupported variable: " + name);
            };
        }
    }

    private record UnaryNode(char operator, Node operand) implements Node {
        @Override
        public double evaluate(long tokens, long hitCacheTokens) {
            double value = operand.evaluate(tokens, hitCacheTokens);
            return operator == '-' ? -value : value;
        }
    }

    private record BinaryNode(char operator, Node left, Node right) implements Node {
        @Override
        public double evaluate(long tokens, long hitCacheTokens) {
            double leftValue = left.evaluate(tokens, hitCacheTokens);
            double rightValue = right.evaluate(tokens, hitCacheTokens);
            return switch (operator) {
                case '+' -> leftValue + rightValue;
                case '-' -> leftValue - rightValue;
                case '*' -> leftValue * rightValue;
                case '/' -> leftValue / rightValue;
                default -> throw new IllegalArgumentException("Unsupported operator: " + operator);
            };
        }
    }

    private static final class Parser {
        private final String input;
        private int position;

        private Parser(String input) {
            this.input = input;
        }

        private Node parseExpression() {
            Node node = parseTerm();
            while (true) {
                skipWhitespace();
                if (match('+')) {
                    node = new BinaryNode('+', node, parseTerm());
                } else if (match('-')) {
                    node = new BinaryNode('-', node, parseTerm());
                } else {
                    return node;
                }
            }
        }

        private Node parseTerm() {
            Node node = parseUnary();
            while (true) {
                skipWhitespace();
                if (match('*')) {
                    node = new BinaryNode('*', node, parseUnary());
                } else if (match('/')) {
                    node = new BinaryNode('/', node, parseUnary());
                } else {
                    return node;
                }
            }
        }

        private Node parseUnary() {
            skipWhitespace();
            if (match('+')) {
                return new UnaryNode('+', parseUnary());
            }
            if (match('-')) {
                return new UnaryNode('-', parseUnary());
            }
            return parsePrimary();
        }

        private Node parsePrimary() {
            skipWhitespace();
            if (match('(')) {
                Node node = parseExpression();
                skipWhitespace();
                if (!match(')')) {
                    throw error("Expected ')'");
                }
                return node;
            }
            if (hasNext() && (Character.isDigit(peek()) || peek() == '.')) {
                return parseNumber();
            }
            if (hasNext() && (Character.isLetter(peek()) || peek() == '_')) {
                return parseVariable();
            }
            throw error("Expected number, variable, or '('");
        }

        private Node parseNumber() {
            int start = position;
            while (hasNext() && (Character.isDigit(peek()) || peek() == '.')) {
                position++;
            }
            if (hasNext() && (peek() == 'e' || peek() == 'E')) {
                position++;
                if (hasNext() && (peek() == '+' || peek() == '-')) {
                    position++;
                }
                while (hasNext() && Character.isDigit(peek())) {
                    position++;
                }
            }
            try {
                return new ConstantNode(Double.parseDouble(input.substring(start, position)));
            } catch (NumberFormatException e) {
                throw error("Invalid number");
            }
        }

        private Node parseVariable() {
            int start = position;
            while (hasNext() && (Character.isLetterOrDigit(peek()) || peek() == '_')) {
                position++;
            }
            String name = input.substring(start, position);
            if (!isSupportedVariable(name)) {
                throw error("Unsupported variable: " + name);
            }
            return new VariableNode(name);
        }

        private boolean isSupportedVariable(String name) {
            return switch (name) {
                case "tokens", "inputLength", "input_length",
                     "hitCacheTokens", "prefixLength", "hit_cache_tokens", "prefix_length" -> true;
                default -> false;
            };
        }

        private void expectEnd() {
            skipWhitespace();
            if (hasNext()) {
                throw error("Unexpected token");
            }
        }

        private boolean match(char expected) {
            if (hasNext() && peek() == expected) {
                position++;
                return true;
            }
            return false;
        }

        private char peek() {
            return input.charAt(position);
        }

        private boolean hasNext() {
            return position < input.length();
        }

        private void skipWhitespace() {
            while (hasNext() && Character.isWhitespace(peek())) {
                position++;
            }
        }

        private IllegalArgumentException error(String message) {
            return new IllegalArgumentException(message + " at position " + position + " in formula: " + input);
        }
    }
}
