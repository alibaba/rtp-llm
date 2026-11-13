package org.flexlb.util.formula;

/**
 * Immutable syntax, exact constant folding, and the small interpreter used for oversized formulas.
 */
final class ArithmeticFormulaAst {
    private ArithmeticFormulaAst() {}

    static Node unary(char op, Node operand) {
        Node node = new UnaryNode(op, operand);
        return operand instanceof ConstantNode ? constant(node) : node;
    }

    static Node binary(char op, Node left, Node right) {
        Node node = new BinaryNode(op, left, right);
        return left instanceof ConstantNode && right instanceof ConstantNode
                ? constant(node)
                : node;
    }

    static Node function(Function function, Node left, Node right) {
        Node node = new FunctionNode(function, left, right);
        return left instanceof ConstantNode && (right == null || right instanceof ConstantNode)
                ? constant(node)
                : node;
    }

    /**
     * Pure evaluation for parameter initializers, constant folding and the JVM size-limit fallback.
     */
    static double evaluate(Node node, double[] vars) {
        return switch (node) {
            case ConstantNode constant -> constant.value();
            case VariableNode variable -> vars[variable.varIndex()];
            case UnaryNode unary -> {
                double value = evaluate(unary.operand(), vars);
                yield unary.op() == '-' ? -value : value;
            }
            case BinaryNode binary -> {
                double left = evaluate(binary.left(), vars);
                double right = evaluate(binary.right(), vars);
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
                double left = evaluate(function.left(), vars);
                double right = function.right() == null ? 0.0 : evaluate(function.right(), vars);
                yield function.function().apply(left, right);
            }
        };
    }

    private static Node constant(Node node) {
        return new ConstantNode(evaluate(node, null));
    }

    /**
     * Marks an expression node consumed by formula validation and compilation.
     */
    sealed interface Node {}

    record ConstantNode(double value) implements Node {}

    record VariableNode(int varIndex) implements Node {}

    record UnaryNode(char op, Node operand) implements Node {}

    record BinaryNode(char op, Node left, Node right) implements Node {}

    record FunctionNode(Function function, Node left, Node right) implements Node {}

    /**
     * Identifies the arithmetic functions accepted by the expression language.
     */
    enum Function {
        SQRT("sqrt", 1),
        LOG("log", 1),
        EXP("exp", 1),
        ABS("abs", 1),
        MAX("max", 2),
        MIN("min", 2),
        POW("pow", 2);

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
}
