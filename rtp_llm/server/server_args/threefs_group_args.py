import argparse
import logging
import os

from rtp_llm.server.server_args.util import str2bool


def init_threefs_group_args(parser):
    ##############################################################################################################
    # 3FS 配置
    ##############################################################################################################
    threefs_group = parser.add_argument_group("3FS")
