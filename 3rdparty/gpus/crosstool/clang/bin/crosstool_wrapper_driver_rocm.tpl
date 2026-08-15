#!/opt/conda310/bin/python
"""Crosstool wrapper for compiling ROCm programs.

SYNOPSIS:
  crosstool_wrapper_driver_rocm [options passed in by cc_library()
                                or cc_binary() rule]

DESCRIPTION:
  This script is expected to be called by the cc_library() or cc_binary() bazel
  rules. When the option "-x rocm" is present in the list of arguments passed
  to this script, it invokes the hipcc compiler. Most arguments are passed
  as is as a string to --compiler-options of hipcc. When "-x rocm" is not
  present, this wrapper invokes gcc with the input arguments as is.
"""

from __future__ import print_function

__author__ = 'whchung@gmail.com (Wen-Heng (Jack) Chung)'

from argparse import ArgumentParser
import os
import subprocess
import re
import sys
from shlex import quote

# Template values set by rocm_configure.bzl.
CPU_COMPILER = ('%{cpu_compiler}')
GCC_HOST_COMPILER_PATH = ('%{gcc_host_compiler_path}')

HIPCC_PATH = '%{hipcc_path}'
PREFIX_DIR = os.path.dirname(GCC_HOST_COMPILER_PATH)
HIPCC_ENV = '%{hipcc_env}'
HIPCC_IS_HIPCLANG = '%{hipcc_is_hipclang}'=="True"
HIP_RUNTIME_PATH = '%{hip_runtime_path}'
HIP_RUNTIME_LIBRARY = '%{hip_runtime_library}'
ROCR_RUNTIME_PATH = '%{rocr_runtime_path}'
ROCR_RUNTIME_LIBRARY = '%{rocr_runtime_library}'
VERBOSE = '%{crosstool_verbose}'=='1'

def Log(s):
  print('gpus/crosstool: {0}'.format(s))


def GetOptionValue(argv, option):
  """Extract the list of values for option from the argv list.

  Args:
    argv: A list of strings, possibly the argv passed to main().
    option: The option whose value to extract, without the leading '-'.

  Returns:
    A list of values, either directly following the option,
    (eg., -opt val1 val2) or values collected from multiple occurrences of
    the option (eg., -opt val1 -opt val2).
  """

  parser = ArgumentParser()
  parser.add_argument('-' + option, nargs='*', action='append')
  args, _ = parser.parse_known_args(argv)
  if not args or not vars(args)[option]:
    return []
  else:
    return sum(vars(args)[option], [])


def ScanIncludeFlags(argv):
  """Collect -I/-isystem/-iquote paths from argv, both attached and separated.

  argparse cannot do this reliably: with nargs='*' a value that itself starts
  with '-' (Bazel emits virtual-include paths as attached -Ipath right after an
  -iquote pair) is either swallowed as the previous flag's value or dropped
  entirely, which silently deletes every _virtual_includes path — the compiler
  then cannot find headers such as alog/Appender.h.

  Returns (isystem, iquote, include) lists, order preserved, duplicates kept
  (they are harmless and preserving order keeps search semantics intact).
  """
  flags = {'-isystem': [], '-iquote': [], '-I': []}
  i = 0
  while i < len(argv):
    tok = argv[i]
    for opt in ('-isystem', '-iquote', '-I'):
      if tok == opt:
        nxt = argv[i + 1] if i + 1 < len(argv) else ''
        # The value itself starts with '-' => upstream squeezed two flags together; do not
        # swallow it: leave it for the next iteration as an independent flag.
        if nxt and not nxt.startswith('-'):
          flags[opt].append(nxt)
          i += 1
        break
      if tok.startswith(opt) and len(tok) > len(opt):
        flags[opt].append(tok[len(opt):])
        break
    i += 1
  return flags['-isystem'], flags['-iquote'], flags['-I']


def GetHostCompilerOptions(argv):
  """Collect the -isystem, -iquote, and --sysroot option values from argv.

  Args:
    argv: A list of strings, possibly the argv passed to main().

  Returns:
    The string that can be used as the --compiler-options to hipcc.
  """

  parser = ArgumentParser()
  parser.add_argument('--sysroot', nargs=1)
  parser.add_argument('-g', nargs='*', action='append')
  parser.add_argument('-fno-canonical-system-headers', action='store_true')
  parser.add_argument('-no-canonical-prefixes', action='store_true')

  args, _ = parser.parse_known_args(argv)

  isystem, iquote, _ = ScanIncludeFlags(argv)

  opts = ''

  for path in isystem:
    opts += ' -isystem ' + quote(path)
  for path in iquote:
    opts += ' -iquote ' + quote(path)
  if args.g:
    opts += ' -g' + ' -g'.join(sum(args.g, []))
  if args.fno_canonical_system_headers:
   opts += ' -no-canonical-prefixes'
  if args.sysroot:
    opts += ' --sysroot ' + args.sysroot[0]

  return opts


def GetHipccOptions(argv):
  """Collect the -hipcc_options values from argv.

  Args:
    argv: A list of strings, possibly the argv passed to main().

  Returns:
    The string that can be passed directly to hipcc.
  """

  parser = ArgumentParser()
  parser.add_argument('-hipcc_options', nargs='*', action='append')

  args, _ = parser.parse_known_args(argv)

  if args.hipcc_options:
    options = _update_options(sum(args.hipcc_options, []))
    return ' '.join(['--'+a for a in options])
  return ''


def InvokeHipcc(argv, log=False):
  """Call hipcc with arguments assembled from argv.

  Args:
    argv: A list of strings, possibly the argv passed to main().
    log: True if logging is requested.

  Returns:
    The return value of calling os.system('hipcc ' + args)
  """

  host_compiler_options = GetHostCompilerOptions(argv)
  hipcc_compiler_options = GetHipccOptions(argv)
  opt_option = GetOptionValue(argv, 'O')
  m_options = GetOptionValue(argv, 'm')
  m_options = ''.join([' -m' + m for m in m_options if m in ['32', '64']])
  _, _, include_options = ScanIncludeFlags(argv)
  out_file = GetOptionValue(argv, 'o')
  depfiles = GetOptionValue(argv, 'MF')
  defines = GetOptionValue(argv, 'D')
  defines = ''.join([' -D' + define for define in defines])
  undefines = GetOptionValue(argv, 'U')
  undefines = ''.join([' -U' + define for define in undefines])
  std_options = GetOptionValue(argv, 'std') + ["c++20"]
  hipcc_allowed_std_options = ["c++11", "c++14", "c++17", "c++20"]
  std_options = ''.join([' -std=' + define
      for define in std_options if define in hipcc_allowed_std_options])

  # The list of source files get passed after the -c option. I don't know of
  # any other reliable way to just get the list of source files to be compiled.
  src_files = GetOptionValue(argv, 'c')

  if len(src_files) == 0:
    return 1
  if len(out_file) != 1:
    return 1

  opt = (' -O2' if (len(opt_option) > 0 and int(opt_option[0]) > 0)
         else ' -g')

  includes = (' -I ' + ' -I '.join(quote(p) for p in include_options)
              if len(include_options) > 0
              else '')

  # Unfortunately, there are other options that have -c prefix too.
  # So allowing only those look like C/C++ files.
  src_files = [f for f in src_files if
               re.search(r'\.cpp$|\.cc$|\.c$|\.cxx$|\.C$|\.cu$', f)]
  srcs = ' '.join(quote(f) for f in src_files)
  out = ' -o ' + quote(out_file[0])

  hipccopts = ' '
  # In hip-clang environment, we need to make sure that hip header is included
  # before some standard math header like <complex> is included in any source.
  # Otherwise, we get build error.
  # Also we need to retain warning about uninitialised shared variable as
  # warning only, even when -Werror option is specified.
  if HIPCC_IS_HIPCLANG:
    hipccopts += ' --include=hip/hip_runtime.h '
  hipccopts += ' ' + hipcc_compiler_options
  # Use -fno-gpu-rdc by default for early GPU kernel finalization
  # This flag would trigger GPU kernels be generated at compile time, instead
  # of link time. This allows the default host compiler (gcc) be used as the
  # linker for TensorFlow on ROCm platform.
  hipccopts += ' -fno-gpu-rdc '
  hipccopts += ' -Wno-unused-command-line-argument '
  hipccopts += undefines
  hipccopts += defines
  hipccopts += std_options
  hipccopts += m_options

  if depfiles:
    # Generate the dependency file
    depfile = quote(depfiles[0])
    cmd = (HIPCC_PATH + ' ' + hipccopts +
           host_compiler_options +
           ' ' + GCC_HOST_COMPILER_PATH +
           ' -I .' + includes + ' ' + srcs + ' -M -o ' + depfile)
    cmd = HIPCC_ENV.replace(';', ' ') + ' ' + cmd
    if log: Log(cmd)
    if VERBOSE: print(cmd)
    exit_status = os.system(cmd)
    if exit_status != 0:
      # On failure, print the raw argv Bazel handed to this wrapper: on CI, "header not found"
      # is often an include flag lost in the wrapping/forwarding stage, and the composed
      # command line alone does not reveal the origin.
      sys.stderr.write('gpus/crosstool: wrapper argv: %s\n' % ' '.join(sys.argv[1:]))
      sys.stderr.write('gpus/crosstool: composed cmd: %s\n' % cmd)
      return exit_status

  cmd = (HIPCC_PATH + ' ' + hipccopts +
         host_compiler_options + ' -fPIC' +
         ' ' + GCC_HOST_COMPILER_PATH +
         ' -I .' + opt + includes + ' -c ' + srcs + out)

  # TODO(zhengxq): for some reason, 'gcc' needs this help to find 'as'.
  # Need to investigate and fix.
  cmd = 'PATH=' + PREFIX_DIR + ':$PATH '\
        + HIPCC_ENV.replace(';', ' ') + ' '\
        + cmd
  if log: Log(cmd)
  if VERBOSE: print(cmd)
  return os.system(cmd)


def main():
  # ignore PWD env var
  os.environ['PWD']=''

  parser = ArgumentParser()
  parser.add_argument('-x', nargs=1)
  parser.add_argument('--rocm_log', action='store_true')
  parser.add_argument('-pass-exit-codes', action='store_true')
  args, leftover = parser.parse_known_args(sys.argv[1:])

  if VERBOSE: print('PWD=' + os.getcwd())
  if VERBOSE: print('HIPCC_ENV=' + HIPCC_ENV)

  if args.x and args.x[0] == 'rocm':
    # compilation for GPU objects
    if args.rocm_log: Log('-x rocm')
    # No wholesale shell quoting here: Bzlmod canonical repo names contain '~', and
    # shlex.quote would wrap the whole token in quotes, turning "-Ipath" into "'-Ipath'" —
    # it no longer looks like a flag (the parser misses it) and gets swallowed as the value of
    # the preceding -iquote, so every _virtual_includes path is lost and the compiler cannot
    # find external repo headers (alog/Appender.h). WORKSPACE naming has no
    # '~', so this defect was invisible before Bzlmod. Quoting is done per value when
    # assembling the command line instead.
    if args.rocm_log: Log('using hipcc')
    return InvokeHipcc(leftover, log=args.rocm_log)

  elif args.pass_exit_codes:
    # link
    # with hipcc compiler invoked with -fno-gpu-rdc by default now, it's ok to
    # use host compiler as linker, but we have to link with HCC/HIP runtime.
    # Such restriction would be revised further as the bazel script get
    # improved to fine tune dependencies to ROCm libraries.
    gpu_linker_flags = [flag for flag in sys.argv[1:]
                               if not flag.startswith(('--rocm_log'))]

    gpu_linker_flags.append('-L' + ROCR_RUNTIME_PATH)
    gpu_linker_flags.append('-Wl,-rpath=' + ROCR_RUNTIME_PATH)
    gpu_linker_flags.append('-l' + ROCR_RUNTIME_LIBRARY)
    gpu_linker_flags.append('-L' + HIP_RUNTIME_PATH)
    gpu_linker_flags.append('-Wl,-rpath=' + HIP_RUNTIME_PATH)
    gpu_linker_flags.append('-l' + HIP_RUNTIME_LIBRARY)
    if HIPCC_IS_HIPCLANG:
      gpu_linker_flags.append("-lrt")

    if VERBOSE: print(' '.join([CPU_COMPILER] + gpu_linker_flags))
    return subprocess.call([CPU_COMPILER] + gpu_linker_flags)

  else:
    # compilation for host objects

    # Strip our flags before passing through to the CPU compiler for files which
    # are not -x rocm. We can't just pass 'leftover' because it also strips -x.
    # We not only want to pass -x to the CPU compiler, but also keep it in its
    # relative location in the argv list (the compiler is actually sensitive to
    # this).
    cpu_compiler_flags = [flag for flag in sys.argv[1:]
                               if not flag.startswith(('--rocm_log'))]

    # XXX: SE codes need to be built with gcc, but need this macro defined
    cpu_compiler_flags.append("-D__HIP_PLATFORM_AMD__")
    if VERBOSE: print(' '.join([CPU_COMPILER] + cpu_compiler_flags))
    return subprocess.call([CPU_COMPILER] + cpu_compiler_flags)

if __name__ == '__main__':
  sys.exit(main())
