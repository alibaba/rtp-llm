package org.flexlb.balance.prediction;

import org.objectweb.asm.ClassTooLargeException;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.Label;
import org.objectweb.asm.MethodTooLargeException;
import org.objectweb.asm.MethodVisitor;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

import static org.flexlb.balance.prediction.ArithmeticFormula.Executable;
import static org.flexlb.balance.prediction.ArithmeticFormulaAst.*;
import static org.objectweb.asm.Opcodes.*;

/** JVM backend: fused aggregates and scoped common subexpressions, without reassociation. */
final class ArithmeticFormulaCompiler {
    private static final String CLASS_NAME = "org/flexlb/balance/prediction/ArithmeticFormula$Generated";
    private static final String SIGNATURE = "([DLjava/util/List;)D";
    private final MethodVisitor method;
    private int nextLocal = 3; // this, scalar variables, batch items

    private ArithmeticFormulaCompiler(MethodVisitor method) {
        this.method = method;
    }

    static Executable compile(Node root) {
        try {
            ClassWriter writer = new ClassWriter(ClassWriter.COMPUTE_FRAMES | ClassWriter.COMPUTE_MAXS);
            writer.visit(V17, ACC_FINAL | ACC_SUPER, CLASS_NAME, null, "java/lang/Object",
                    new String[]{Executable.class.getName().replace('.', '/')});
            MethodVisitor init = writer.visitMethod(ACC_PUBLIC, "<init>", "()V", null, null);
            init.visitCode();
            init.visitVarInsn(ALOAD, 0);
            init.visitMethodInsn(INVOKESPECIAL, "java/lang/Object", "<init>", "()V", false);
            init.visitInsn(RETURN);
            init.visitMaxs(0, 0);
            init.visitEnd();
            MethodVisitor method = writer.visitMethod(ACC_PUBLIC, "evaluate", SIGNATURE, null, null);
            method.visitCode();
            new ArithmeticFormulaCompiler(method).expression(root);
            method.visitInsn(DRETURN);
            method.visitMaxs(0, 0);
            method.visitEnd();
            writer.visitEnd();
            var lookup = MethodHandles.lookup().defineHiddenClass(writer.toByteArray(), true);
            return (Executable) lookup.findConstructor(lookup.lookupClass(), MethodType.methodType(void.class))
                    .invoke();
        } catch (MethodTooLargeException | ClassTooLargeException tooLarge) {
            // Preserve support for expressions exceeding JVM class/method limits.
            return (vars, items) -> evaluate(root, vars, items);
        } catch (Throwable failure) {
            if (failure instanceof Error error) throw error;
            throw new IllegalStateException("Cannot compile arithmetic formula", failure);
        }
    }

    private void expression(Node root) {
        Scope outer = new Scope(1, 2);
        Map<AggregateFuncNode, Integer> aggregates = new LinkedHashMap<>();
        collectAggregates(root, aggregates, new HashSet<>());
        if (aggregates.size() > 1) {
            fuseAggregates(aggregates, outer);
        }
        emit(root, outer);
    }

    private static void collectAggregates(Node node, Map<AggregateFuncNode, Integer> aggregates,
                                          Set<Node> visited) {
        if (!visited.add(node)) return;
        switch (node) {
            // Nested sums belong to this item's scalar scope, not the batch.
            case AggregateFuncNode aggregate -> aggregates.putIfAbsent(aggregate, 0);
            case UnaryNode unary -> collectAggregates(unary.operand(), aggregates, visited);
            case BinaryNode binary -> {
                collectAggregates(binary.left(), aggregates, visited);
                collectAggregates(binary.right(), aggregates, visited);
            }
            case FunctionNode function -> {
                collectAggregates(function.left(), aggregates, visited);
                if (function.right() != null) collectAggregates(function.right(), aggregates, visited);
            }
            default -> { }
        }
    }

    /** One traversal, independent ordered accumulators, shared per-item expressions. */
    private void fuseAggregates(Map<AggregateFuncNode, Integer> aggregates, Scope outer) {
        aggregates.replaceAll((aggregate, ignored) -> {
            int slot = nextLocal;
            nextLocal += 2;
            outer.locals.put(aggregate, slot);
            return slot;
        });
        Label scalar = new Label(), end = new Label(), loop = new Label(), done = new Label();
        method.visitVarInsn(ALOAD, outer.items);
        method.visitJumpInsn(IFNULL, scalar);
        method.visitVarInsn(ALOAD, outer.items);
        method.visitMethodInsn(INVOKEINTERFACE, "java/util/List", "isEmpty", "()Z", true);
        method.visitJumpInsn(IFNE, scalar);
        for (int slot : aggregates.values()) {
            method.visitInsn(DCONST_0);
            method.visitVarInsn(DSTORE, slot);
        }
        int iterator = nextLocal++, vars = nextLocal++;
        method.visitVarInsn(ALOAD, outer.items);
        method.visitMethodInsn(INVOKEINTERFACE, "java/util/List", "iterator", "()Ljava/util/Iterator;", true);
        method.visitVarInsn(ASTORE, iterator);
        method.visitLabel(loop);
        method.visitVarInsn(ALOAD, iterator);
        method.visitMethodInsn(INVOKEINTERFACE, "java/util/Iterator", "hasNext", "()Z", true);
        method.visitJumpInsn(IFEQ, done);
        method.visitVarInsn(ALOAD, iterator);
        method.visitMethodInsn(INVOKEINTERFACE, "java/util/Iterator", "next", "()Ljava/lang/Object;", true);
        method.visitTypeInsn(CHECKCAST, "[D");
        method.visitVarInsn(ASTORE, vars);
        Scope itemScope = new Scope(vars, -1);
        for (var entry : aggregates.entrySet()) {
            method.visitVarInsn(DLOAD, entry.getValue());
            emit(entry.getKey().arg(), itemScope);
            method.visitInsn(DADD);
            method.visitVarInsn(DSTORE, entry.getValue());
        }
        method.visitJumpInsn(GOTO, loop);
        method.visitLabel(done);
        method.visitJumpInsn(GOTO, end);
        method.visitLabel(scalar);
        Scope scalarScope = new Scope(outer.vars, -1);
        for (var entry : aggregates.entrySet()) {
            emit(entry.getKey().arg(), scalarScope);
            method.visitVarInsn(DSTORE, entry.getValue());
        }
        method.visitLabel(end);
    }

    private void emit(Node node, Scope scope) {
        Integer existing = scope.locals.get(node);
        if (existing != null) {
            method.visitVarInsn(DLOAD, existing);
            return;
        }
        switch (node) {
            case ConstantNode constant -> method.visitLdcInsn(constant.value());
            case VariableNode variable -> {
                method.visitVarInsn(ALOAD, scope.vars);
                method.visitLdcInsn(variable.varIndex());
                method.visitInsn(DALOAD);
            }
            case UnaryNode unary -> {
                emit(unary.operand(), scope);
                if (unary.op() == '-') method.visitInsn(DNEG);
            }
            case BinaryNode binary -> {
                emit(binary.left(), scope);
                emit(binary.right(), scope);
                if (binary.op() == '^') {
                    math("pow", 2);
                } else {
                    method.visitInsn(switch (binary.op()) {
                        case '+' -> DADD;
                        case '-' -> DSUB;
                        case '*' -> DMUL;
                        case '/' -> DDIV;
                        default -> throw new IllegalStateException("Unknown operator");
                    });
                }
            }
            case FunctionNode function -> {
                emit(function.left(), scope);
                if (function.right() != null) emit(function.right(), scope);
                math(function.function().mathName, function.function().arity);
            }
            case AggregateFuncNode aggregate -> sum(aggregate.arg(), scope);
        }
        int slot = nextLocal;
        nextLocal += 2;
        method.visitInsn(DUP2);
        method.visitVarInsn(DSTORE, slot);
        scope.locals.put(node, slot);
    }

    private void sum(Node item, Scope outer) {
        if (outer.items < 0) {
            // Nested sum sees a scalar item, exactly like the interpreter.
            emit(item, outer);
            return;
        }
        AggregateFuncNode aggregate = new AggregateFuncNode(item);
        Map<AggregateFuncNode, Integer> aggregates = new LinkedHashMap<>();
        aggregates.put(aggregate, 0);
        fuseAggregates(aggregates, outer);
        method.visitVarInsn(DLOAD, aggregates.get(aggregate));
    }

    private void math(String name, int arity) {
        method.visitMethodInsn(INVOKESTATIC, "java/lang/Math", name, arity == 1 ? "(D)D" : "(DD)D", false);
    }

    private static final class Scope {
        private final int vars;
        private final int items;
        private final Map<Node, Integer> locals = new HashMap<>();

        private Scope(int vars, int items) {
            this.vars = vars;
            this.items = items;
        }
    }
}
