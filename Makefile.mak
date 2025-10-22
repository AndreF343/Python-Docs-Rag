# Makefile — simple, shell-agnostic targets
PY := python
VENV := .venv

.PHONY: help init collect analyze chunk clean

help:
	@echo "Targets:"
	@echo "  make init      - create venv and install requirements"
	@echo "  make collect   - run scripts/collect_from_indexes.py"
	@echo "  make analyze   - run scripts/analyze_corpus.py"
	@echo "  make chunk     - run scripts/chunk_corpus.py"
	@echo "  make clean     - remove derived artifacts"

init:
	python -m venv $(VENV)
	. $(VENV)/bin/activate; pip install -r requirements.txt

collect:
	. $(VENV)/bin/activate; $(PY) scripts/collect_from_indexes.py

analyze:
	. $(VENV)/bin/activate; $(PY) scripts/analyze_corpus.py

chunk:
	. $(VENV)/bin/activate; $(PY) scripts/chunk_corpus.py

clean:
	rm -rf data/analysis stats.json data/chunks/*.jsonl

index:
	. .venv/bin/activate; python scripts/build_index.py

search:
	. .venv/bin/activate; python scripts/search_index.py "how to open files" --k 5

eval:
	. .venv/bin/activate; python scripts/eval_retrieval.py
