# Documentation Manual
## Prepare Docs Requirements
``` 
wget --no-check-certificate https://github.com/jgm/pandoc/releases/download/3.7.0.2/pandoc-3.7.0.2-linux-amd64.tar.gz -O pandoc-3.7.0.2-linux-amd64.tar.gz &&     tar xzf pandoc-3.7.0.2-linux-amd64.tar.gz &&     cp pandoc-3.7.0.2/bin/pandoc /usr/local/bin/ &&     rm -rf pandoc-3.7.0.2*/opt/conda310/bin/pip install uv 

/opt/conda310/bin/uv pip install sphinx myst-nb jupyter nbconvert sphinx-autobuild --cache-dir=/root/.cache/uv/      --python=/opt/conda310/bin/python      --verbose

yum update -y &&     yum install -y moreutils --skip-broken || true &&     yum install -y parallel --skip-broken || true &&     rm -rf /var/cache/yum/*

/opt/conda310/bin/uv pip install -r  docs/requirements.txt       --cache-dir=/root/.cache/uv/      --python=/opt/conda310/bin/python      --verbose
```

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