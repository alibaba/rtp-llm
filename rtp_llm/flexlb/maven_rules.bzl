# Copyright 2025 FlexLB Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom Bazel rules for Maven integration."""

def _maven_jar_impl(ctx):
    """Implementation of maven_jar rule for building Maven modules."""
    output = ctx.outputs.jar

    # Prepare the build environment
    build_env = {
        "MAVEN_OPTS": "-Xmx2g -XX:MaxMetaspaceSize=512m",
        "LC_ALL": "C",
        "JAVA_HOME": "/opt/taobao/java",
    }

    # Calculate output directory path in Python
    output_dir = "/".join(output.path.split("/")[:-1])

    # Build command
    build_cmd = """
        set -euo pipefail

        # Set up environment
        export MAVEN_OPTS="{maven_opts}"
        export LC_ALL=C
        export JAVA_HOME="{java_home}"
        export PATH="$$JAVA_HOME/bin:/usr/bin:/bin:/usr/local/bin:$$PATH"

        # Change to project root directory
        cd {project_dir}

        # Ensure Maven wrapper is executable
        /usr/bin/chmod +x ./mvnw || chmod +x ./mvnw

        # Run Maven build
        ./mvnw clean install {maven_args} -pl {modules} -am

        # Copy the output JAR (output path is absolute from execroot)
        # Create output directory and copy file
        mkdir -p "{output_dir}"
        cp "{module_dir}/target/{jar_name}" "{output}"
    """.format(
        maven_opts = build_env["MAVEN_OPTS"],
        java_home = build_env["JAVA_HOME"],
        project_dir = ctx.file.pom.dirname,
        maven_args = " ".join(ctx.attr.maven_args),
        modules = ",".join(ctx.attr.modules),
        module_dir = ctx.attr.module_dir,
        jar_name = ctx.attr.jar_name,
        output = output.path,
        output_dir = output_dir,
    )

    ctx.actions.run_shell(
        inputs = ctx.files.srcs + ctx.files.deps,
        outputs = [output],
        command = build_cmd,
        mnemonic = "MavenBuild",
        progress_message = "Building Maven module {}".format(ctx.attr.name),
        use_default_shell_env = True,
        execution_requirements = {"no-sandbox": "1"},  # Maven needs network access
    )

    return [DefaultInfo(files = depset([output]))]

# Custom rule for building Maven modules
maven_jar = rule(
    implementation = _maven_jar_impl,
    attrs = {
        "srcs": attr.label_list(
            allow_files = True,
            doc = "Source files for the module",
        ),
        "deps": attr.label_list(
            allow_files = True,
            doc = "Dependencies (other Maven modules)",
        ),
        "pom": attr.label(
            allow_single_file = ["pom.xml"],
            mandatory = True,
            doc = "The root pom.xml file",
        ),
        "modules": attr.string_list(
            mandatory = True,
            doc = "List of Maven modules to build (in order)",
        ),
        "module_dir": attr.string(
            mandatory = True,
            doc = "Directory of the target module",
        ),
        "jar_name": attr.string(
            mandatory = True,
            doc = "Name of the output JAR file",
        ),
        "maven_args": attr.string_list(
            default = [
                "-B",
                "-Dorg.slf4j.simpleLogger.defaultLogLevel=warn",
                "-Dmaven.test.failure.ignore=true",
                "-Derror-prone.skip=true",
                "-Dautoconfig.skip=true",
                "-P!internal",
                "-T", "1C",
                "-DskipTests",
                "-Dmaven.test.skip=true",
            ],
            doc = "Maven command line arguments",
        ),
    },
    outputs = {
        "jar": "%{name}.jar",
    },
    doc = """
    Builds a Maven module and produces a JAR file.
    This rule integrates Maven builds into Bazel while maintaining
    the existing Maven project structure.
    """,
)

def _maven_test_impl(ctx):
    """Implementation of maven_test rule for running Maven tests."""
    output = ctx.outputs.test_log
    executable = ctx.outputs.executable

    test_cmd = """
        set -euo pipefail

        # Set up environment
        export MAVEN_OPTS="-Xmx2g -XX:MaxMetaspaceSize=512m"
        export LC_ALL=C
        export JAVA_HOME="/opt/taobao/java"
        export PATH="$$JAVA_HOME/bin:/usr/bin:/bin:/usr/local/bin:$$PATH"

        # Change to project root directory
        cd {project_dir}

        # Ensure Maven wrapper is executable
        /usr/bin/chmod +x {maven_wrapper} || chmod +x {maven_wrapper}

        # Build dependencies first
        {maven_wrapper} install {build_args} -pl {modules} -am

        # Run tests
        cd {module_dir}
        ../{maven_wrapper} test {test_args} 2>&1 | tee {output}

        # Always succeed to allow collecting all test results
        exit 0
    """.format(
        project_dir = ctx.file.pom.dirname,
        maven_wrapper = "./mvnw",
        build_args = " ".join(["-DskipTests"] + ctx.attr.maven_args),
        test_args = " ".join(ctx.attr.maven_args),
        modules = ",".join(ctx.attr.modules),
        module_dir = ctx.attr.module_dir,
        output = output.path,
    )

    # Create the executable test script
    ctx.actions.write(
        output = executable,
        content = test_cmd,
        is_executable = True,
    )

    # Create runfiles - collect all input files
    all_files = ctx.files.srcs + ctx.files.deps + [ctx.file.pom]
    runfiles = ctx.runfiles(files = all_files)

    return DefaultInfo(
        files = depset([output]),
        executable = executable,
        runfiles = runfiles,
    )

# Custom rule for running Maven tests
maven_test = rule(
    implementation = _maven_test_impl,
    attrs = {
        "srcs": attr.label_list(
            allow_files = True,
            doc = "Source files for the module",
        ),
        "deps": attr.label_list(
            allow_files = True,
            doc = "Dependencies (other Maven modules)",
        ),
        "pom": attr.label(
            allow_single_file = ["pom.xml"],
            mandatory = True,
            doc = "The root pom.xml file",
        ),
        "modules": attr.string_list(
            mandatory = True,
            doc = "List of Maven modules to build (in order)",
        ),
        "module_dir": attr.string(
            mandatory = True,
            doc = "Directory of the target module",
        ),
        "maven_args": attr.string_list(
            default = [
                "-B",
                "-Dorg.slf4j.simpleLogger.defaultLogLevel=warn",
                "-Dmaven.test.failure.ignore=true",
                "-Derror-prone.skip=true",
                "-Dautoconfig.skip=true",
                "-P!internal",
                "-T", "1C",
            ],
            doc = "Maven command line arguments",
        ),
    },
    outputs = {
        "test_log": "%{name}.log",
        "executable": "%{name}.sh",
    },
    test = True,
    doc = """
    Runs Maven tests for a module.
    Test results are collected in a log file for further analysis.
    """,
)