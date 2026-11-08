import org.eclipse.jdt.core.dom.ASTNode;
import org.eclipse.jdt.core.dom.ASTVisitor;
import org.eclipse.jdt.core.dom.AnonymousClassDeclaration;
import org.eclipse.jdt.core.dom.Comment;
import org.eclipse.jdt.core.dom.CompilationUnit;
import org.eclipse.jdt.core.dom.EnumDeclaration;
import org.eclipse.jdt.core.dom.ImportDeclaration;
import org.eclipse.jdt.core.dom.MethodDeclaration;
import org.eclipse.jdt.core.dom.MethodInvocation;
import org.eclipse.jdt.core.dom.Name;
import org.eclipse.jdt.core.dom.RecordDeclaration;
import org.eclipse.jdt.core.dom.SimpleName;
import org.eclipse.jdt.core.dom.TypeDeclaration;
import org.eclipse.text.edits.ReplaceEdit;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * Converts changed JUnit assertion calls without resolving the project's compile classpath.
 * Ambiguous names retain their qualifier so formatting cannot change method selection.
 */
final class TestAssertionImports {
    private static final String ASSERTIONS = "org.junit.jupiter.api.Assertions";

    record Result(String source, TreeSet<Integer> changedLines) {
    }

    static Result update(String source, TreeSet<Integer> changed, CompilationUnit unit) {
        List<ImportDeclaration> imports = new ArrayList<>();
        Set<String> blockedMethods = new HashSet<>();
        boolean importsAssertions = false;
        boolean importsJUnitPackage = false;
        boolean hasOtherAssertionsImport = false;
        boolean hasOtherPackageWildcard = false;
        boolean importsAllAssertions = false;
        for (Object entry : unit.imports()) {
            ImportDeclaration declaration = (ImportDeclaration) entry;
            imports.add(declaration);
            String name = declaration.getName().getFullyQualifiedName();
            if (!declaration.isStatic()) {
                importsAssertions |= !declaration.isOnDemand() && name.equals(ASSERTIONS);
                importsJUnitPackage |= declaration.isOnDemand() && name.equals("org.junit.jupiter.api");
                hasOtherPackageWildcard |= declaration.isOnDemand() && !name.equals("org.junit.jupiter.api");
                hasOtherAssertionsImport |=
                        !declaration.isOnDemand() && name.endsWith(".Assertions") && !name.equals(ASSERTIONS);
            } else if (declaration.isOnDemand()) {
                if (!name.equals(ASSERTIONS)) {
                    return new Result(source, changed);
                }
                importsAllAssertions = true;
            } else if (!name.startsWith(ASSERTIONS + ".")) {
                blockedMethods.add(name.substring(name.lastIndexOf('.') + 1));
            }
        }
        NamesInFile names = new NamesInFile();
        unit.accept(names);
        if (names.hasInheritedMethods) {
            return new Result(source, changed);
        }
        blockedMethods.addAll(names.methodNames);
        boolean shortNameAllowed =
                !names.shadowsAssertions && !hasOtherAssertionsImport && !blockedMethods.contains("Assertions")
                        && (importsAssertions || importsJUnitPackage && !hasOtherPackageWildcard);
        List<ReplaceEdit> edits = new ArrayList<>();
        Set<String> requiredImports = new TreeSet<>();
        TreeSet<Integer> affectedLines = new TreeSet<>(changed);
        unit.accept(new ASTVisitor() {
            @Override
            public boolean visit(MethodInvocation call) {
                String method = call.getName().getIdentifier();
                if (!(call.getExpression() instanceof Name receiver) || !call.typeArguments().isEmpty()
                        || blockedMethods.contains(method)) {
                    return true;
                }
                String receiverName = receiver.getFullyQualifiedName();
                if (receiverName.equals(ASSERTIONS) && (names.shadowsJUnitPackage || blockedMethods.contains("org"))) {
                    return true;
                }
                if (!receiverName.equals(ASSERTIONS) && !(shortNameAllowed && receiverName.equals("Assertions"))) {
                    return true;
                }
                int firstLine = unit.getLineNumber(call.getStartPosition());
                int lastLine = unit.getLineNumber(call.getStartPosition() + call.getLength() - 1);
                Integer nextChanged = changed.ceiling(firstLine);
                if (nextChanged == null || nextChanged > lastLine) {
                    return true;
                }
                int start = receiver.getStartPosition();
                int length = call.getName().getStartPosition() - start;
                // Keep comments, line breaks and explicit type arguments in qualified invocations intact.
                if (!source.substring(start, start + length).matches("[\\w.$ \\t]+")) {
                    return true;
                }
                edits.add(new ReplaceEdit(start, length, ""));
                requiredImports.add(ASSERTIONS + "." + method);
                affectedLines.add(firstLine);
                return true;
            }
        });
        if (edits.isEmpty()) {
            return new Result(source, changed);
        }
        if (!usesAssertionsName(unit, edits)) {
            for (ImportDeclaration declaration : imports) {
                if (!declaration.isStatic() && !declaration.isOnDemand()
                        && declaration.getName().getFullyQualifiedName().equals(ASSERTIONS)) {
                    edits.add(removeImport(source, unit, declaration));
                }
            }
        }
        if (!importsAllAssertions) {
            addImports(source, unit, imports, requiredImports, edits);
        }
        return rewrite(source, unit, affectedLines, edits);
    }

    /**
     * Import edits shift source lines; carry the original formatting selection to the updated source.
     */
    private static Result rewrite(String source,
                                  CompilationUnit unit,
                                  TreeSet<Integer> affectedLines,
                                  List<ReplaceEdit> edits) {
        TreeSet<Integer> shiftedLines = new TreeSet<>();
        for (int line : affectedLines) {
            int position = unit.getPosition(line, 0);
            int end = afterLine(source, position);
            int shifted = line;
            int addedWithinLine = 0;
            for (ReplaceEdit edit : edits) {
                int delta = newlines(edit.getText())
                        - newlines(source.substring(edit.getOffset(), edit.getOffset() + edit.getLength()));
                if (edit.getOffset() <= position) {
                    shifted += delta;
                } else if (edit.getOffset() < end) {
                    addedWithinLine += delta;
                }
            }
            for (int current = Math.max(1, shifted); current <= Math.max(1, shifted + addedWithinLine); current++) {
                shiftedLines.add(current);
            }
        }
        StringBuilder updated = new StringBuilder(source);
        edits.sort(Comparator.comparingInt(ReplaceEdit::getOffset).thenComparingInt(ReplaceEdit::getLength).reversed());
        for (ReplaceEdit edit : edits) {
            updated.replace(edit.getOffset(), edit.getOffset() + edit.getLength(), edit.getText());
        }
        return new Result(updated.toString(), shiftedLines);
    }

    private static boolean usesAssertionsName(CompilationUnit unit, List<ReplaceEdit> removals) {
        List<SimpleName> remaining = new ArrayList<>();
        unit.accept(new ASTVisitor(true) {
            @Override
            public boolean visit(ImportDeclaration declaration) {
                return false;
            }

            @Override
            public boolean visit(SimpleName name) {
                if (name.getIdentifier().equals("Assertions") && removals.stream()
                        .noneMatch(edit -> name.getStartPosition() >= edit.getOffset()
                                && name.getStartPosition() < edit.getOffset() + edit.getLength())) {
                    remaining.add(name);
                }
                return true;
            }
        });
        return !remaining.isEmpty();
    }

    private static ReplaceEdit removeImport(String source, CompilationUnit unit, ImportDeclaration declaration) {
        int start = declaration.getStartPosition();
        int end = start + declaration.getLength();
        int lineStart = unit.getPosition(unit.getLineNumber(start), 0);
        int lineEnd = afterLine(source, end);
        if (source.substring(lineStart, start).isBlank() && source.substring(end, lineEnd).isBlank()) {
            return new ReplaceEdit(lineStart, lineEnd - lineStart, "");
        }
        return new ReplaceEdit(start, end - start, "");
    }

    private static void addImports(String source,
                                   CompilationUnit unit,
                                   List<ImportDeclaration> imports,
                                   Set<String> required,
                                   List<ReplaceEdit> edits) {
        List<ImportDeclaration> statics = imports.stream().filter(ImportDeclaration::isStatic).toList();
        Set<String> existing = new HashSet<>();
        for (ImportDeclaration declaration : statics) {
            existing.add(declaration.getName().getFullyQualifiedName());
        }
        TreeMap<Integer, List<String>> insertions = new TreeMap<>();
        for (String name : required) {
            if (existing.contains(name)) {
                continue;
            }
            ASTNode previous =
                    !statics.isEmpty() ? statics.getLast() : !imports.isEmpty() ? imports.getLast() : unit.getPackage();
            int position = previous == null ? 0 : afterDeclaration(source, unit, previous);
            for (ImportDeclaration declaration : statics) {
                if (declaration.getName().getFullyQualifiedName().compareTo(name) > 0) {
                    int start = declaration.getStartPosition();
                    int lineStart = unit.getPosition(unit.getLineNumber(start), 0);
                    position = source.substring(lineStart, start).isBlank() ? lineStart : start;
                    break;
                }
            }
            insertions.computeIfAbsent(position, ignored -> new ArrayList<>()).add("import static " + name + ";");
        }
        for (var insertion : insertions.entrySet()) {
            int position = insertion.getKey();
            String prefix = position > 0 && (statics.isEmpty() || source.charAt(position - 1) != '\n') ? "\n" : "";
            String suffix = position == 0 ? "\n\n" : "\n";
            edits.add(new ReplaceEdit(position, 0, prefix + String.join("\n", insertion.getValue()) + suffix));
        }
    }

    private static int afterDeclaration(String source, CompilationUnit unit, ASTNode declaration) {
        int end = declaration.getStartPosition() + declaration.getLength();
        int lineEnd = afterLine(source, end);
        for (Object entry : unit.getCommentList()) {
            Comment comment = (Comment) entry;
            if (comment.getStartPosition() < lineEnd && comment.getStartPosition() + comment.getLength() > lineEnd) {
                return end;
            }
        }
        List<?> following =
                declaration == unit.getPackage() && !unit.imports().isEmpty() ? unit.imports() : unit.types();
        for (Object entry : following) {
            int start = ((ASTNode) entry).getStartPosition();
            if (start >= end && start < lineEnd) {
                return end;
            }
        }
        return lineEnd;
    }

    private static int afterLine(String source, int position) {
        int newline = source.indexOf('\n', position);
        return newline < 0 ? source.length() : newline + 1;
    }

    private static int newlines(String text) {
        return (int) text.chars().filter(character -> character == '\n').count();
    }

    /**
     * Without type bindings, local declarations and inherited methods can hide statically imported assertions.
     */
    private static final class NamesInFile extends ASTVisitor {
        private final Set<String> methodNames = new HashSet<>();
        private boolean shadowsAssertions;
        private boolean shadowsJUnitPackage;
        private boolean hasInheritedMethods;

        @Override
        public boolean visit(SimpleName name) {
            shadowsAssertions |= name.isDeclaration() && name.getIdentifier().equals("Assertions");
            shadowsJUnitPackage |= name.isDeclaration() && name.getIdentifier().equals("org");
            return true;
        }

        @Override
        public boolean visit(MethodDeclaration method) {
            methodNames.add(method.getName().getIdentifier());
            return true;
        }

        @Override
        public boolean visit(TypeDeclaration type) {
            hasInheritedMethods |= type.getSuperclassType() != null || !type.superInterfaceTypes().isEmpty();
            return true;
        }

        @Override
        public boolean visit(EnumDeclaration type) {
            hasInheritedMethods |= !type.superInterfaceTypes().isEmpty();
            return true;
        }

        @Override
        public boolean visit(RecordDeclaration type) {
            hasInheritedMethods |= !type.superInterfaceTypes().isEmpty();
            return true;
        }

        @Override
        public boolean visit(AnonymousClassDeclaration type) {
            hasInheritedMethods = true;
            return true;
        }
    }
}
