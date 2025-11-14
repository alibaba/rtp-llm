import os
import sys
from unittest import TestCase, main

print(os.getcwd())
print(
    "PYTHONPATH="
    + os.environ["PYTHONPATH"]
    + " LD_LIBRARY_PATH="
    + os.environ["LD_LIBRARY_PATH"]
    + " "
    + sys.executable
    + " "
)

from rtp_llm.start_server import main as server_main


class ServerTest(TestCase):
    def test_simple(self):
        server_main()


if __name__ == "__main__":
    main()
