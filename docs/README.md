# Documentation Manual
## Prepare Docs Requirements

From the repository root, use an existing RTP-LLM development environment:

```bash
python -m pip install -e '.[docs]'
python -m pip install sphinx-intl sphinx-autobuild
cd docs
```

The `docs` extra in `pyproject.toml` supplies the configured Book theme and
Sphinx extensions. The translation commands below also use `sphinx-intl`.
Notebook pages additionally require the existing Pandoc installation on `PATH`.

## Add Documentation

### 1. Add Documentation File
* For design-related content, add MD (Markdown) files. For usage documentation that can be executed, consider adding Jupyter Notebook (`.ipynb`) files.
* If images are included, it is recommended to place them in the `docs/pics/` directory.

### 2. Add Index
* The documentation index is built from `.rst` files. After adding documentation, organize files into the appropriate directory structure based on their purpose.

### 3. Add Chinese Support
1. Navigate to the `docs` directory.
2. Run the command `make update`. This command extracts all translatable messages and updates the PO translation files.
3. Manually update the newly generated translation files under `docs/locales/zh_CN/LC_MESSAGES`, filling in the `msgstr` fields for the target language.
4. Run `make html` to generate both Chinese and English HTML versions. Alternatively, run `make html-zh` or `make html-en` separately to build specific language versions.

### Modify Documentation
Similar to adding documentation, if you modify existing documentation, ensure to update the corresponding Chinese PO files and regenerate HTML output by running `make html`.
1. Run `make update`, Manually update the newly generated translation files under `docs/locales/zh_CN/LC_MESSAGES`, filling in the `msgstr` fields for the target language.
2. Run `make html` to generate both Chinese and English HTML versions. Alternatively, run `make html-zh` or `make html-en` separately to build specific language versions.
