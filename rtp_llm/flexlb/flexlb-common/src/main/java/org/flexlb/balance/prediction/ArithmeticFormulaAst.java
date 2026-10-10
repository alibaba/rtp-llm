package org.flexlb.balance.prediction;

import java.util.List;

/** Immutable syntax, exact constant folding, and the small interpreter used for oversized formulas. */
final class ArithmeticFormulaAst {
    private ArithmeticFormulaAst() { }

    sealed interface Node { }
    record ConstantNode(double value) implements Node { }
    record VariableNode(int varIndex) implements Node { }
    record UnaryNode(char op, Node operand) implements Node { }
    record BinaryNode(char op, Node left, Node right) implements Node { }
    record FunctionNode(Function function, Node left, Node right) implements Node { }
    record AggregateFuncNode(Node arg) implements Node { }

    enum Function {
        SQRT("sqrt", 1), LOG("log", 1), EXP("exp", 1), ABS("abs", 1),
        MAX("max", 2), MIN("min", 2), POW("pow", 2);

        final String mathName;
        final int arity;

        Function(String mathName, int arity) {
            this.mathName = mathName;
            this.arity = arity;
        }

        static Function named(String name) {
            for (Function function : values()) {
                if (function.mathName.equals(name)) return function;
            }
            return null;
        }

        double apply(double left, double right) {
            return switch (this) {
                case SQRT -> Math.sqrt(left);
                case LOG -> Math.log(left);
                case EXP -> Math.exp(left);
                case ABS -> Math.abs(left);
                case MAX -> Math.max(left, right);
                case MIN -> Math.min(left, right);
                case POW -> Math.pow(left, right);
            };
        }
    }

    static Node unary(char op, Node operand) {
        Node node = new UnaryNode(op, operand);
        return operand instanceof ConstantNode ? constant(node) : node;
    }

    static Node binary(char op, Node left, Node right) {
        Node node = new BinaryNode(op, left, right);
        return left instanceof ConstantNode && right instanceof ConstantNode ? constant(node) : node;
    }

    static Node function(Function function, Node left, Node right) {
        Node node = new FunctionNode(function, left, right);
        return left instanceof ConstantNode && (right == null || right instanceof ConstantNode)
                ? constant(node) : node;
    }

    private static Node constant(Node node) {
        return new ConstantNode(evaluate(node, null, null));
    }

    /** Pure evaluation for parameter initializers, constant folding and the JVM size-limit fallback. */
    static double evaluate(Node node, double[] vars, List<double[]> items) {
        return switch (node) {
            case ConstantNode constant -> constant.value();
            case VariableNode variable -> vars[variable.varIndex()];
            case UnaryNode unary -> {
                double value = evaluate(unary.operand(), vars, items);
                yield unary.op() == '-' ? -value : value;
            }
            case BinaryNode binary -> {
                double left = evaluate(binary.left(), vars, items);
                double right = evaluate(binary.right(), vars, items);
                yield switch (binary.op()) {
                    case '+' -> left + right;
                    case '-' -> left - right;
                    case '*' -> left * right;
                    case '/' -> left / right;
                    case '^' -> Math.pow(left, right);
                    default -> throw new IllegalStateException("Unknown operator: " + binary.op());
                };
            }
            case FunctionNode function -> {
                double left = evaluate(function.left(), vars, items);
                double right = function.right() == null ? 0.0 : evaluate(function.right(), vars, items);
                yield function.function().apply(left, right);
            }
            case AggregateFuncNode aggregate -> {
                if (items == null || items.isEmpty()) yield evaluate(aggregate.arg(), vars, null);
                double total = 0.0;
                for (double[] item : items) total += evaluate(aggregate.arg(), item, null);
                yield total;
            }
        };
    }
}
