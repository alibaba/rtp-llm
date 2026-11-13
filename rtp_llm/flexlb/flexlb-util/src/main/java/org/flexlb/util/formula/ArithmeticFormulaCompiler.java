package org.flexlb.util.formula;

import org.objectweb.asm.ClassTooLargeException;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.MethodTooLargeException;
import org.objectweb.asm.MethodVisitor;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.HashMap;
import java.util.Map;

import static org.flexlb.util.formula.ArithmeticFormula.Executable;
import static org.flexlb.util.formula.ArithmeticFormulaAst.BinaryNode;
import static org.flexlb.util.formula.ArithmeticFormulaAst.ConstantNode;
import static org.flexlb.util.formula.ArithmeticFormulaAst.FunctionNode;
import static org.flexlb.util.formula.ArithmeticFormulaAst.Node;
import static org.flexlb.util.formula.ArithmeticFormulaAst.UnaryNode;
import static org.flexlb.util.formula.ArithmeticFormulaAst.VariableNode;
import static org.flexlb.util.formula.ArithmeticFormulaAst.binary;
import static org.flexlb.util.formula.ArithmeticFormulaAst.evaluate;
import static org.flexlb.util.formula.ArithmeticFormulaAst.function;
import static org.flexlb.util.formula.ArithmeticFormulaAst.unary;
import static org.objectweb.asm.Opcodes.ACC_FINAL;
import static org.objectweb.asm.Opcodes.ACC_PUBLIC;
import static org.objectweb.asm.Opcodes.ACC_SUPER;
import static org.objectweb.asm.Opcodes.ALOAD;
import static org.objectweb.asm.Opcodes.DADD;
import static org.objectweb.asm.Opcodes.DALOAD;
import static org.objectweb.asm.Opcodes.DDIV;
import static org.objectweb.asm.Opcodes.DLOAD;
import static org.objectweb.asm.Opcodes.DMUL;
import static org.objectweb.asm.Opcodes.DNEG;
import static org.objectweb.asm.Opcodes.DRETURN;
import static org.objectweb.asm.Opcodes.DSTORE;
import static org.objectweb.asm.Opcodes.DSUB;
import static org.objectweb.asm.Opcodes.DUP2;
import static org.objectweb.asm.Opcodes.INVOKESPECIAL;
import static org.objectweb.asm.Opcodes.INVOKESTATIC;
import static org.objectweb.asm.Opcodes.RETURN;
import static org.objectweb.asm.Opcodes.V17;

/**
 * Compiles scalar expressions with common subexpressions, without reassociation.
 */
final class ArithmeticFormulaCompiler {
    private static final String CLASS_NAME = "org/flexlb/util/formula/ArithmeticFormula$Generated";
    private static final String SIGNATURE = "([D)D";

    private final MethodVisitor method;

    private int nextLocal = 2; // this, scalar variables

    private ArithmeticFormulaCompiler(MethodVisitor method) {
        this.method = method;
    }

    static Executable compile(Node root) {
        try {
            ClassWriter writer =
                    new ClassWriter(ClassWriter.COMPUTE_FRAMES | ClassWriter.COMPUTE_MAXS);
            writer.visit(
                    V17,
                    ACC_FINAL | ACC_SUPER,
                    CLASS_NAME,
                    null,
                    "java/lang/Object",
                    new String[] {Executable.class.getName().replace('.', '/')});
            MethodVisitor init = writer.visitMethod(ACC_PUBLIC, "<init>", "()V", null, null);
            init.visitCode();
            init.visitVarInsn(ALOAD, 0);
            init.visitMethodInsn(INVOKESPECIAL, "java/lang/Object", "<init>", "()V", false);
            init.visitInsn(RETURN);
            init.visitMaxs(0, 0);
            init.visitEnd();
            MethodVisitor method =
                    writer.visitMethod(ACC_PUBLIC, "evaluate", SIGNATURE, null, null);
            method.visitCode();
            new ArithmeticFormulaCompiler(method).expression(root);
            method.visitInsn(DRETURN);
            method.visitMaxs(0, 0);
            method.visitEnd();
            writer.visitEnd();
            var lookup = MethodHandles.lookup().defineHiddenClass(writer.toByteArray(), true);
            return (Executable)
                    lookup.findConstructor(lookup.lookupClass(), MethodType.methodType(void.class))
                            .invoke();
        } catch (MethodTooLargeException | ClassTooLargeException tooLarge) {
            // Preserve support for expressions exceeding JVM class/method limits.
            return vars -> evaluate(root, vars);
        } catch (Throwable failure) {
            if (failure instanceof Error error) throw error;
            throw new IllegalStateException("Cannot compile arithmetic formula", failure);
        }
    }

    private void expression(Node root) {
        emit(root, new Scope(1));
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
                    method.visitInsn(
                            switch (binary.op()) {
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
        }
        int slot = nextLocal;
        nextLocal += 2;
        method.visitInsn(DUP2);
        method.visitVarInsn(DSTORE, slot);
        scope.locals.put(node, slot);
    }

    private void math(String name, int arity) {
        method.visitMethodInsn(
                INVOKESTATIC, "java/lang/Math", name, arity == 1 ? "(D)D" : "(DD)D", false);
    }

    /**
     * Tracks the input array slot and cached expression-node locals for one bytecode compilation.
     */
    private static final class Scope {
        private final int vars;

        private final Map<Node, Integer> locals = new HashMap<>();

        private Scope(int vars) {
            this.vars = vars;
        }
    }
}
