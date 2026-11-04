import com.sun.source.tree.ClassTree;
import com.sun.source.tree.CompilationUnitTree;
import com.sun.source.tree.MethodTree;
import com.sun.source.tree.Tree;
import com.sun.source.util.JavacTask;
import com.sun.source.util.TreeScanner;
import com.sun.source.util.Trees;

import javax.tools.ToolProvider;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.regex.Pattern;

/** Source budgets, including private methods, overloads and explicitly written constructors. */
public final class DeliveryBudget {
    private static final class Counts {
        int methods;
        long longest;
        String method;
        long effectiveLines, functionLines;
        int parameters, nesting;
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            throw new IllegalArgumentException("Pass delivery source files; Java 21 is required");
        }
        boolean violated = false;
        var compiler = ToolProvider.getSystemJavaCompiler();
        try (var files = compiler.getStandardFileManager(null, null, null)) {
            var task =
                    (JavacTask)
                            compiler.getTask(
                                    null,
                                    files,
                                    null,
                                    List.of("-proc:none"),
                                    null,
                                    files.getJavaFileObjectsFromStrings(List.of(args)));
            var positions = Trees.instance(task).getSourcePositions();
            System.out.println(
                    "file\ttype\tmethods\tmax_function_lines\tlongest_method\tfile_lines\ttype_code_lines\tmax_function_code_lines\tmax_parameters\tmax_nesting");
            for (CompilationUnitTree unit : task.parse()) {
                String source = unit.getSourceFile().getCharContent(true).toString();
                String code = withoutComments(source);
                var owners = new ArrayDeque<String>();
                var counts = new LinkedHashMap<String, Counts>();
                new TreeScanner<Void, Void>() {
                    @Override
                    public Void visitClass(ClassTree node, Void ignored) {
                        String name = node.getSimpleName().toString();
                        owners.addLast(
                                name.isEmpty()
                                        ? "<anonymous@"
                                                + unit.getLineMap()
                                                        .getLineNumber(
                                                                positions.getStartPosition(
                                                                        unit, node))
                                                + ">"
                                        : name);
                        Counts count =
                                counts.computeIfAbsent(
                                        String.join(".", owners), key -> new Counts());
                        count.effectiveLines =
                                effectiveLines(
                                        code,
                                        positions.getStartPosition(unit, node),
                                        positions.getEndPosition(unit, node));
                        super.visitClass(node, ignored);
                        owners.removeLast();
                        return null;
                    }

                    @Override
                    public Void visitMethod(MethodTree node, Void ignored) {
                        long start = positions.getStartPosition(unit, node);
                        long end = positions.getEndPosition(unit, node);
                        if (start >= 0 && end >= start && !owners.isEmpty()) {
                            var count = counts.get(String.join(".", owners));
                            count.methods++;
                            count.functionLines =
                                    Math.max(count.functionLines, effectiveLines(code, start, end));
                            count.parameters =
                                    Math.max(count.parameters, node.getParameters().size());
                            count.nesting = Math.max(count.nesting, nesting(node.getBody()));
                            long lines =
                                    unit.getLineMap().getLineNumber(end - 1)
                                            - unit.getLineMap().getLineNumber(start)
                                            + 1;
                            if (lines > count.longest) {
                                count.longest = lines;
                                count.method = node.getName().toString();
                            }
                        }
                        return super.visitMethod(node, ignored);
                    }
                }.scan(unit, null);
                var path = Path.of(unit.getSourceFile().toUri());
                long fileLines;
                try (var lines = Files.lines(path)) {
                    fileLines = lines.count();
                }
                for (var entry : counts.entrySet()) {
                    var count = entry.getValue();
                    System.out.println(
                            path
                                    + "\t"
                                    + entry.getKey()
                                    + "\t"
                                    + count.methods
                                    + "\t"
                                    + count.longest
                                    + "\t"
                                    + count.method
                                    + "\t"
                                    + fileLines
                                    + "\t"
                                    + count.effectiveLines
                                    + "\t"
                                    + count.functionLines
                                    + "\t"
                                    + count.parameters
                                    + "\t"
                                    + count.nesting);
                    violated |=
                            count.methods > 30
                                    || count.functionLines > 50
                                    || count.effectiveLines > 1000
                                    || count.parameters > 6
                                    || count.nesting > 3;
                }
            }
        }
        if (violated) {
            System.exit(1);
        }
    }

    private static String withoutComments(String source) {
        // Match literals first so comment markers inside strings remain code.
        var tokens =
                Pattern.compile(
                        "\\\"\\\"\\\"[\\s\\S]*?\\\"\\\"\\\"|\\\"(?:\\\\.|[^\\\"\\\\])*\\\"|'(?:\\\\.|[^'\\\\])*'|/\\*[\\s\\S]*?\\*/|//[^\\r\\n]*");
        var matcher = tokens.matcher(source);
        StringBuilder code = new StringBuilder(source);
        while (matcher.find()) {
            if (source.charAt(matcher.start()) != '/') continue;
            for (int i = matcher.start(); i < matcher.end(); i++) {
                if (code.charAt(i) != '\n' && code.charAt(i) != '\r') code.setCharAt(i, ' ');
            }
        }
        return code.toString();
    }

    private static long effectiveLines(String code, long start, long end) {
        if (start < 0 || end <= start) return 0;
        return code.substring((int) start, (int) end)
                .lines()
                .filter(line -> !line.isBlank())
                .count();
    }

    private static int nesting(Tree body) {
        if (body == null) return 0;
        int[] maximum = {0};
        new TreeScanner<Void, Integer>() {
            @Override
            public Void scan(Tree node, Integer depth) {
                if (node == null || node instanceof ClassTree) return null;
                boolean control =
                        switch (node.getKind()) {
                            case IF,
                                    FOR_LOOP,
                                    ENHANCED_FOR_LOOP,
                                    WHILE_LOOP,
                                    DO_WHILE_LOOP,
                                    SWITCH,
                                    SWITCH_EXPRESSION,
                                    TRY,
                                    SYNCHRONIZED ->
                                    true;
                            default -> false;
                        };
                int next = depth + (control ? 1 : 0);
                maximum[0] = Math.max(maximum[0], next);
                return super.scan(node, next);
            }
        }.scan(body, 0);
        return maximum[0];
    }
}
