#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "rtp-llm-private"
copyright = "2023, Advanced Micro Devices, Inc. All rights reserved"
author = "AMD rtp-llm team"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["recommonmark", "sphinx_markdown_tables"]
html_theme = "sphinx_pdj_theme"
html_static_path = ["_static"]

source_suffix = [".rst", ".md"]

def get_log_files():
    log_file_names = []
    for file in os.listdir(os.getcwd()):
        if file.endswith(".log"):
            file_name = os.path.splitext(file)[0]
            log_file_names.append(file_name)
            turn_log_into_md(file, file_name)
    return log_file_names


def turn_log_into_md(log_file, file_name):
    if not os.path.exists("docs"):
        os.makedirs("docs")
    with open(log_file) as log, open(f"docs/{file_name}.md", "w") as md:
        md.write(f"# {file_name}\n")
        md.write("<pre>\n")
        for line in log:
            # md.write(line)
            md.write(html.escape(line))
        md.write("</pre>\n")    


log_file_names = get_log_files()
html_context = {
    "log_files": log_file_names,
}