import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import javax.tools.*;

class CompilePinnedSources {
    public static void main(String[] args) throws Exception {
        String sha = args[0];
        var compiler = ToolProvider.getSystemJavaCompiler();
        var sources = new ArrayList<JavaFileObject>();
        for (int i = 4; i < args.length; i++) {
            String name = args[i];
            var git = new ProcessBuilder("git", "show", sha + ":" + name).start();
            String source = new String(git.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            if (git.waitFor() != 0) throw new IllegalStateException(name);
            sources.add(
                    new SimpleJavaFileObject(
                            URI.create("string:///" + name), JavaFileObject.Kind.SOURCE) {
                        public CharSequence getCharContent(boolean ignoreEncodingErrors) {
                            return source;
                        }
                    });
            System.out.println("Compiled from commit " + sha + " " + name);
        }
        var options =
                List.of(
                        "--release",
                        "21",
                        "-classpath",
                        args[1],
                        "-processorpath",
                        args[2],
                        "-d",
                        args[3]);
        if (!compiler.getTask(null, null, null, options, null, sources).call()) System.exit(1);
    }
}
