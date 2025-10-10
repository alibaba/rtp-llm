import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../.."))

version_file = "../rtp_llm/release_version.py"
with open(version_file, "r") as f:
    exec(compile(f.read(), version_file, "exec"))
__version__ = locals()["RELEASE_VERSION"]

project = "RTP-LLM"
copyright = f"2023-{datetime.now().year}, RTP-LLM"
author = "RTP-LLM Team"

version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx_tabs.tabs",
    "myst_parser",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
    "nbsphinx",
    "sphinx.ext.mathjax",
]

# 国际化配置
locale_dirs = ["locales/"]  # 翻译文件存放目录
gettext_compact = False  # 为每个文档生成单独的 .pot 文件

nbsphinx_allow_errors = True
nbsphinx_execute = "never"

autosectionlabel_prefix_document = True
nbsphinx_allow_directives = True

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
    "html_image",
    "linkify",
    "substitution",
]

myst_heading_anchors = 3

nbsphinx_kernel_name = "python3"
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

nb_render_priority = {
    "html": (
        "application/vnd.jupyter.widget-view+json",
        "application/javascript",
        "text/html",
        "image/svg+xml",
        "image/png",
        "image/jpeg",
        "text/markdown",
        "text/latex",
        "text/plain",
    )
}

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
    "html_image",
    "linkify",
    "substitution",
]

myst_heading_anchors = 3
myst_ref_domains = ["std", "py"]

templates_path = ["_templates"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

language = "en"

exclude_patterns = [
    "build",
    "Thumbs.db",
    ".DS_Store",
    "index",
    "assets",
    "index.html",
]

pygments_style = "sphinx"

html_theme = "sphinx_book_theme"
html_logo = "_static/image/logo.png"
html_favicon = "_static/image/logo.ico"
html_title = project
html_copy_source = True
html_last_updated_fmt = ""

html_theme_options = {
    "repository_url": "https://github.com/alibaba/rtp-llm",
    "repository_branch": "main",
    "show_navbar_depth": 3,
    "max_navbar_depth": 4,
    "collapse_navbar": True,
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "use_sidenotes": True,
    "show_toc_level": 2,
}

html_context = {
    "display_github": True,
    "github_user": "alibaba",
    "github_repo": "rtp-llm",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

html_static_path = ["_static"]
html_css_files = ["css/custom_log.css", "css/language-switcher.css"]
html_js_files = ["js/language-switcher.js"]


def setup(app):
    app.add_css_file("css/custom_log.css")
    app.add_css_file("css/language-switcher.css")
    app.add_js_file("js/language-switcher.js")


myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
]
myst_heading_anchors = 5

htmlhelp_basename = "RTP-LLMdoc"

latex_elements = {}

latex_documents = [
    (master_doc, "RTP-LLM.tex", "RTP-LLM Documentation", "RTP-LLM Team", "manual"),
]

man_pages = [(master_doc, "RTP-LLM", "RTP-LLM Documentation", [author], 1)]

texinfo_documents = [
    (
        master_doc,
        "RTP-LLM",
        "RTP-LLM Documentation",
        author,
        "RTP-LLM",
        "One line description of project.",
        "Miscellaneous",
    ),
]

epub_title = project

epub_exclude_files = ["search.html"]

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

autodoc_preserve_defaults = True
navigation_with_keys = False

autodoc_mock_imports = [
    "torch",
    "transformers",
    "triton",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.12", None),
    "typing_extensions": ("https://typing-extensions.readthedocs.io/en/latest", None),
    "pillow": ("https://pillow.readthedocs.io/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

html_theme = "sphinx_book_theme"

nbsphinx_prolog = """
.. raw:: html

    <style>
        .output_area.stderr, .output_area.stdout {
            color: #d3d3d3 !important; /* light gray */
        }
    </style>
"""
