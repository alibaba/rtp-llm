import org.eclipse.jdt.core.JavaCore;
import org.eclipse.jdt.core.ToolFactory;
import org.eclipse.jdt.core.dom.AST;
import org.eclipse.jdt.core.dom.ASTNode;
import org.eclipse.jdt.core.dom.ASTParser;
import org.eclipse.jdt.core.dom.ASTVisitor;
import org.eclipse.jdt.core.dom.AbstractTypeDeclaration;
import org.eclipse.jdt.core.dom.AnonymousClassDeclaration;
import org.eclipse.jdt.core.dom.Block;
import org.eclipse.jdt.core.dom.BodyDeclaration;
import org.eclipse.jdt.core.dom.Comment;
import org.eclipse.jdt.core.dom.CompilationUnit;
import org.eclipse.jdt.core.dom.MethodDeclaration;
import org.eclipse.jdt.core.dom.NodeFinder;
import org.eclipse.jdt.core.dom.Statement;
import org.eclipse.jdt.core.dom.SwitchCase;
import org.eclipse.jdt.core.formatter.CodeFormatter;
import org.eclipse.jface.text.Document;
import org.eclipse.jface.text.IRegion;
import org.eclipse.jface.text.Region;
import org.eclipse.text.edits.MultiTextEdit;
import org.eclipse.text.edits.ReplaceEdit;
import org.eclipse.text.edits.TextEdit;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Formats only the changed lines of Java files with the FlexLB Eclipse formatter profile.
 * Usage: FormatChangedLines <formatter.xml> <file>:<first-last>[,<first-last>...]|all ...
 * A changed line is widened to the whole statement or declaration header that contains it, so wrapped
 * parameters and conditions are aligned as one unit; block bodies below that header stay as written.
 * A changed one-line Javadoc such as "/** text *&#47;" is expanded to the multi-line form.
 */
public class FormatChangedLines {
    private static final Pattern SETTING = Pattern.compile("<setting id=\"([^\"]+)\" value=\"([^\"]*)\"");
    private static final Pattern ONE_LINE_JAVADOC =
            Pattern.compile("^([ \\t]*)/\\*\\* ([^*\\n]*[^* \\n]) \\*/[ \\t]*$");

    public static void main(String[] args) throws Exception {
        Map<String, String> options = readOptions(Path.of(args[0]));
        CodeFormatter formatter = ToolFactory.createCodeFormatter(options);
        for (int i = 1; i < args.length; i++) {
            int separator = args[i].lastIndexOf(':');
            Path file = Path.of(args[i].substring(0, separator));
            String source = Files.readString(file);
            List<String> lines = new ArrayList<>(List.of(source.split("\n", -1)));
            TreeSet<Integer> changed = changedLines(args[i].substring(separator + 1), lines.size());
            changed = expandJavadoc(lines, changed);
            source = String.join("\n", lines);
            String formatted = format(formatter, options, source, changed);
            if (!formatted.equals(Files.readString(file))) {
                Files.writeString(file, formatted);
                System.out.println("formatted " + file);
            }
        }
    }

    private static Map<String, String> readOptions(Path profile) throws Exception {
        Map<String, String> options = new HashMap<>();
        Matcher matcher = SETTING.matcher(Files.readString(profile));
        while (matcher.find()) {
            options.put(matcher.group(1), matcher.group(2));
        }
        return options;
    }

    /**
     * Parses "3-5,9-9" into 1-based line numbers; "all" selects every line.
     */
    private static TreeSet<Integer> changedLines(String ranges, int lineCount) {
        TreeSet<Integer> changed = new TreeSet<>();
        if (ranges.equals("all")) {
            for (int line = 1; line <= lineCount; line++) {
                changed.add(line);
            }
            return changed;
        }
        for (String range : ranges.split(",")) {
            String[] bounds = range.split("-");
            for (int line = Integer.parseInt(bounds[0]); line <= Integer.parseInt(bounds[1]); line++) {
                changed.add(line);
            }
        }
        return changed;
    }

    /**
     * Expands changed one-line Javadoc comments in place and returns the changed line numbers after expansion.
     */
    private static TreeSet<Integer> expandJavadoc(List<String> lines, TreeSet<Integer> changed) {
        TreeSet<Integer> shifted = new TreeSet<>();
        int offset = 0;
        for (int line = 1; line <= lines.size() - offset; line++) {
            int index = line - 1 + offset;
            if (!changed.contains(line)) {
                continue;
            }
            Matcher matcher = ONE_LINE_JAVADOC.matcher(lines.get(index));
            if (matcher.matches()) {
                String indent = matcher.group(1);
                lines.set(index, indent + "/**");
                lines.add(index + 1, indent + " * " + matcher.group(2));
                lines.add(index + 2, indent + " */");
                shifted.add(index + 1);
                shifted.add(index + 2);
                shifted.add(index + 3);
                offset += 2;
            } else {
                shifted.add(index + 1);
            }
        }
        return shifted;
    }

    private static String format(CodeFormatter formatter,
                                 Map<String, String> options,
                                 String source,
                                 TreeSet<Integer> changed)
            throws Exception {
        CompilationUnit unit = parse(source, options);
        TreeSet<Integer> lines = new TreeSet<>();
        for (int line : changed) {
            widen(unit, source, line, lines);
        }
        List<IRegion> regions = new ArrayList<>();
        Integer first = null;
        Integer previous = null;
        for (int line : lines) {
            if (previous != null && line != previous + 1) {
                regions.add(lineRegion(unit, source, first, previous));
                first = null;
            }
            if (first == null) {
                first = line;
            }
            previous = line;
        }
        if (first != null) {
            regions.add(lineRegion(unit, source, first, previous));
        }
        if (regions.isEmpty()) {
            return source;
        }
        TextEdit fullFileEdit = formatter.format(CodeFormatter.K_COMPILATION_UNIT, source, 0, source.length(), 0, "\n");
        if (fullFileEdit == null) {
            throw new IllegalStateException("Eclipse formatter could not parse the file");
        }
        MultiTextEdit changedRegionEdit = new MultiTextEdit();
        for (ReplaceEdit edit : replaceEdits(fullFileEdit)) {
            if (endsInside(edit, regions)) {
                changedRegionEdit.addChild(new ReplaceEdit(edit.getOffset(), edit.getLength(), edit.getText()));
            }
        }
        Document document = new Document(source);
        changedRegionEdit.apply(document);
        return document.get();
    }

    /**
     * Eclipse reports formatting as whitespace replacements. Formatting the whole file and keeping the
     * replacements that end inside a changed region gives each changed line the indentation it has in the
     * fully formatted file, while lines outside the regions keep their bytes. The replacement that ends at
     * the first token of a region carries that line's indentation; the one that starts on the last line of
     * a region and ends on the next line belongs to the next line and is skipped.
     */
    private static boolean endsInside(ReplaceEdit edit, List<IRegion> regions) {
        int end = edit.getOffset() + edit.getLength();
        for (IRegion region : regions) {
            if (end > region.getOffset() && end <= region.getOffset() + region.getLength()) {
                return true;
            }
        }
        return false;
    }

    private static List<ReplaceEdit> replaceEdits(TextEdit edit) {
        List<ReplaceEdit> edits = new ArrayList<>();
        if (edit instanceof ReplaceEdit replace) {
            edits.add(replace);
        }
        for (TextEdit child : edit.getChildren()) {
            edits.addAll(replaceEdits(child));
        }
        return edits;
    }

    private static CompilationUnit parse(String source, Map<String, String> options) {
        ASTParser parser = ASTParser.newParser(AST.getJLSLatest());
        Map<String, String> compilerOptions = JavaCore.getOptions();
        JavaCore.setComplianceOptions(options.get(JavaCore.COMPILER_SOURCE), compilerOptions);
        parser.setCompilerOptions(compilerOptions);
        parser.setKind(ASTParser.K_COMPILATION_UNIT);
        parser.setSource(source.toCharArray());
        return (CompilationUnit) parser.createAST(null);
    }

    /**
     * Adds the lines of the statement or declaration header that contains the changed line.
     * Blank lines, brace-only lines and comment lines are formatted on their own.
     */
    private static void widen(CompilationUnit unit, String source, int line, TreeSet<Integer> lines) {
        int lineStart = unit.getPosition(line, 0);
        if (lineStart < 0) {
            return;
        }
        int lineEnd = source.indexOf('\n', lineStart);
        String text = source.substring(lineStart, lineEnd < 0 ? source.length() : lineEnd).strip();
        if (text.isEmpty() || text.matches("[{}();,\\s]*")) {
            return;
        }
        int codeStart = lineStart + source.substring(lineStart).indexOf(text);
        if (insideComment(unit, codeStart)) {
            lines.add(line);
            return;
        }
        ASTNode node = NodeFinder.perform(unit, codeStart, 0);
        while (node != null && !(node instanceof Statement && !(node instanceof Block))
                && !(node instanceof BodyDeclaration)) {
            node = node.getParent();
        }
        if (node == null) {
            lines.add(line);
            return;
        }
        addHeader(unit, source, node, line, lines);
        ASTNode method = node.getParent();
        while (method != null && !(method instanceof MethodDeclaration)) {
            method = method.getParent();
        }
        if (method != null) {
            addHeader(unit, source, method, line, lines);
        }
    }

    private static void addHeader(CompilationUnit unit, String source, ASTNode node, int line, TreeSet<Integer> lines) {
        int startLine = unit.getLineNumber(node.getStartPosition());
        int endLine = unit.getLineNumber(headerEnd(node, source));
        if (line > endLine && !(node instanceof MethodDeclaration)) {
            endLine = line;
        }
        for (int header = Math.min(startLine, line); header <= endLine; header++) {
            lines.add(header);
        }
    }

    /**
     * End of the node's header: the opening brace of its first nested body, or the node end when it has none.
     */
    private static int headerEnd(ASTNode node, String source) {
        int[] firstBody = {node.getStartPosition() + node.getLength() - 1};
        if (node instanceof AbstractTypeDeclaration type) {
            firstBody[0] = source.indexOf('{', type.getName().getStartPosition());
            return firstBody[0];
        }
        node.accept(new ASTVisitor() {
            @Override
            public boolean preVisit2(ASTNode child) {
                if (child != node && (child instanceof Block || child instanceof AnonymousClassDeclaration
                        || child instanceof SwitchCase)) {
                    firstBody[0] = Math.min(firstBody[0], child.getStartPosition());
                    return false;
                }
                return true;
            }
        });
        return firstBody[0];
    }

    private static boolean insideComment(CompilationUnit unit, int position) {
        for (Object comment : unit.getCommentList()) {
            Comment c = (Comment) comment;
            if (position >= c.getStartPosition() && position < c.getStartPosition() + c.getLength()) {
                return true;
            }
        }
        return false;
    }

    private static IRegion lineRegion(CompilationUnit unit, String source, int firstLine, int lastLine) {
        int start = unit.getPosition(firstLine, 0);
        int end = lastLine < unit.getLineNumber(source.length() - 1)
                ? unit.getPosition(lastLine + 1, 0) - 1
                : source.length();
        return new Region(start, end - start);
    }
}
