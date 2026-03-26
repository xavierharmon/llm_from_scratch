# Makefile — convenience commands for llm_from_scratch
# Usage: make <target>

.PHONY: setup data train test clean demo help

# ── Setup ──────────────────────────────────────────────────────────────────
setup:
	pip install -r requirements.txt
	@echo "\n✓ Dependencies installed"

# ── Data pipeline ──────────────────────────────────────────────────────────
data:
	python 01_data/download_data.py --source synthetic --n 10000
	python 01_data/preprocessing.py
	python 02_tokenization/train_tokenizer.py --vocab-size 8000
	@echo "\n✓ Data pipeline complete"

# ── Training ───────────────────────────────────────────────────────────────
train:
	python 06_training/train.py --config configs/small.yaml

train-debug:
	python 06_training/train.py --config configs/debug.yaml

train-medium:
	python 06_training/train.py --config configs/medium.yaml

# ── Tests ──────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/ -v --tb=short -x -q

test-all:
	pytest tests/ 02_tokenization/tests/ -v --tb=short

# ── Demo ───────────────────────────────────────────────────────────────────
demo:
	@if [ ! -f experiments/baseline/best.pt ]; then \
		echo "No checkpoint found. Run 'make train' first."; \
		exit 1; \
	fi
	python 07_inference/demo.py --checkpoint experiments/baseline/best.pt

# ── Scaling laws ───────────────────────────────────────────────────────────
scaling:
	python 10_scaling/scaling_laws.py

# ── Clean ──────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned"

clean-experiments:
	rm -rf experiments/baseline experiments/debug
	@echo "✓ Experiments cleaned"

# ── Help ───────────────────────────────────────────────────────────────────
help:
	@echo "llm_from_scratch — available commands:"
	@echo ""
	@echo "  make setup          Install all dependencies"
	@echo "  make data           Generate data + train tokenizer (full pipeline)"
	@echo "  make train          Train small model (~2hrs on CPU)"
	@echo "  make train-debug    Train tiny model (seconds, for testing)"
	@echo "  make train-medium   Train medium model (GPU recommended)"
	@echo "  make test           Run all tests"
	@echo "  make demo           Launch interactive generation demo"
	@echo "  make scaling        Print scaling law table"
	@echo "  make clean          Remove __pycache__ and .pyc files"
	@echo ""
	@echo "  Quick start:"
	@echo "    make setup && make data && make train-debug && make demo"
