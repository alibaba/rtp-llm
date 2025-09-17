# Minimal Makefile for Sphinx documentation
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAUTOBUILD ?= sphinx-autobuild
SOURCEDIR     = .
BUILDDIR      = build
PORT          ?= 8003
PYTHON        ?= python3

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo ""
	@echo "Additional targets:"
	@echo "  serve       to build and serve documentation with auto-build and live reload"
	@echo "  html        to build multi-language HTML documentation (en and zh_CN)"
	@echo "  html-en     to build English HTML documentation only"
	@echo "  html-zh     to build Chinese HTML documentation only"
	@echo "  update      to update translation files (extract messages and update PO files)"

# Build multi-language HTML documentation
html:
	@echo "Building multi-language HTML documentation..."
	@$(PYTHON) build_multilang.py

# Build English HTML documentation only
html-en:
	@echo "Building English HTML documentation..."
	@$(PYTHON) build_multilang.py en

# Build Chinese HTML documentation only
html-zh:
	@echo "Building Chinese HTML documentation..."
	@$(PYTHON) build_multilang.py zh_CN

# Update translation files (extract messages and update PO files)
update:
	@echo "Updating translation files..."
	@$(PYTHON) build_multilang.py update

# Compile Notebook files and record execution time
compile:
	@set -e; \
	echo "Starting Notebook compilation..."; \
	mkdir -p logs; \
	echo "Notebook execution timings:" > logs/timing.log; \
	START_TOTAL=$$(date +%s); \
	find $(SOURCEDIR) -path "*/build/*" -prune -o -name "*.ipynb" -print0 | \
		parallel -0 -j3 --halt soon,fail=1 ' \
		NB_NAME=$$(basename {}); \
		START_TIME=$$(date +%s); \
		retry --delay=0 --times=2 -- \
			jupyter nbconvert --to notebook --execute --inplace "{}" \
			--ExecutePreprocessor.timeout=600 \
			--ExecutePreprocessor.kernel_name=python3; \
		RET_CODE=$$?; \
		END_TIME=$$(date +%s); \
		ELAPSED_TIME=$$((END_TIME - START_TIME)); \
		echo "$${NB_NAME}: $${ELAPSED_TIME}s" >> logs/timing.log; \
		exit $$RET_CODE' || exit 1; \
	END_TOTAL=$$(date +%s); \
	TOTAL_ELAPSED=$$((END_TOTAL - START_TOTAL)); \
	echo "---------------------------------" >> logs/timing.log; \
	echo "Total execution time: $${TOTAL_ELAPSED}s" >> logs/timing.log; \
	echo "All Notebook execution timings:" && cat logs/timing.log

# Serve documentation with auto-build and live reload
serve:
	@echo "Starting auto-build server at http://localhost:$(PORT)"
	@$(SPHINXAUTOBUILD) "$(SOURCEDIR)" "$(BUILDDIR)/html" \
                --host 0.0.0.0 \
		--port $(PORT) \
		--watch $(SOURCEDIR) \
		--re-ignore ".*\.(ipynb_checkpoints|pyc|pyo|pyd|git)"

.PHONY: help Makefile compile clean serve html html-en html-zh update

# Override the default html target to use our multi-language build
# This prevents the default Sphinx html target from being used
html-single:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	find . -name "*.ipynb" -exec nbstripout {} \;
	rm -rf $(BUILDDIR)
	rm -rf logs
	@$(PYTHON) build_multilang.py clean

# Catch-all target for other Sphinx targets (but not html)
%: Makefile
	@if [ "$@" = "html" ]; then \
		$(MAKE) html; \
	else \
		$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O); \
	fi