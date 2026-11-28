"""Select checkpoint test legs from the Bazel platform, not installed packages."""

import os
import unittest


def has_native_fst():
    # Standalone invocations keep exercising FST unless explicitly on a
    # wheel-less lock. A missing wheel on a supported lock must fail the test.
    return os.environ.get("NEWLOADER_TEST_NATIVE_FST") != "0"


def checkpoint_readers(scratch, fst):
    return (scratch, fst) if has_native_fst() else (scratch,)


requires_native_fst = unittest.skipUnless(has_native_fst(), "platform lock has no native FST reader")
