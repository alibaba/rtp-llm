PYTHON ?= python3
LATEXMK ?= latexmk
TECTONIC ?= tectonic
ARTIFACT_ARGS ?=
PAPER := NaviSched-State-Aware-Cost-Optimization-for-Distributed-LLM-Prefill-Scheduling

.PHONY: all artifacts clean

all: $(PAPER).pdf

artifacts:
	$(PYTHON) generate_artifacts.py $(ARTIFACT_ARGS)

$(PAPER).pdf: $(PAPER).tex references.bib artifacts
	@if command -v $(LATEXMK) >/dev/null 2>&1; then \
		$(LATEXMK) -pdf -interaction=nonstopmode $(PAPER).tex; \
	elif command -v $(TECTONIC) >/dev/null 2>&1; then \
		$(TECTONIC) $(PAPER).tex; \
	else \
		echo "latexmk or tectonic is required" >&2; \
		exit 1; \
	fi

clean:
	@if command -v $(LATEXMK) >/dev/null 2>&1; then \
		$(LATEXMK) -C $(PAPER).tex; \
	else \
		$(RM) $(PAPER).pdf; \
	fi
