import com.sun.source.tree.ClassTree;
import com.sun.source.tree.CompilationUnitTree;
import com.sun.source.tree.MethodTree;
import com.sun.source.util.JavacTask;
import com.sun.source.util.TreeScanner;
import com.sun.source.util.Trees;
import javax.tools.ToolProvider;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.LinkedHashMap;
import java.util.List;

/** Source budgets, including private methods, overloads and explicitly written constructors. */
public final class DeliveryBudget {
    private static final class Counts {
        int methods;
        long longest;
        String method;
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 0) { throw new IllegalArgumentException("Pass delivery source files; Java 21 is required"); }
        boolean violated = false;
        var compiler = ToolProvider.getSystemJavaCompiler();
        try (var files = compiler.getStandardFileManager(null, null, null)) {
            var task = (JavacTask) compiler.getTask(null, files, null, List.of("-proc:none"), null,
                    files.getJavaFileObjectsFromStrings(List.of(args)));
            var positions = Trees.instance(task).getSourcePositions();
            System.out.println("file\ttype\tmethods\tmax_function_lines\tlongest_method\tfile_lines");
            for (CompilationUnitTree unit : task.parse()) {
                var owners = new ArrayDeque<String>();
                var counts = new LinkedHashMap<String, Counts>();
                new TreeScanner<Void, Void>() {
                    @Override public Void visitClass(ClassTree node, Void ignored) {
                        String name = node.getSimpleName().toString();
                        owners.addLast(name.isEmpty() ? "<anonymous@" + unit.getLineMap().getLineNumber(positions.getStartPosition(unit, node)) + ">" : name);
                        counts.computeIfAbsent(String.join(".", owners), key -> new Counts());
                        super.visitClass(node, ignored);
                        owners.removeLast();
                        return null;
                    }

                    @Override public Void visitMethod(MethodTree node, Void ignored) {
                        long start = positions.getStartPosition(unit, node);
                        long end = positions.getEndPosition(unit, node);
                        if (start >= 0 && end >= start && !owners.isEmpty()) {
                            var count = counts.get(String.join(".", owners));
                            count.methods++;
                            long lines = unit.getLineMap().getLineNumber(end - 1)
                                    - unit.getLineMap().getLineNumber(start) + 1;
                            if (lines > count.longest) { count.longest = lines; count.method = node.getName().toString(); }
                        }
                        return super.visitMethod(node, ignored);
                    }
                }.scan(unit, null);
                var path = Path.of(unit.getSourceFile().toUri());
                long fileLines;
                try (var lines = Files.lines(path)) { fileLines = lines.count(); }
                for (var entry : counts.entrySet()) {
                    var count = entry.getValue();
                    System.out.println(path + "\t" + entry.getKey() + "\t" + count.methods + "\t"
                            + count.longest + "\t" + count.method + "\t" + fileLines);
                    violated |= count.methods > 30 || count.longest > 100;
                }
            }
        }
        if (violated) { System.exit(1); }
    }
}
