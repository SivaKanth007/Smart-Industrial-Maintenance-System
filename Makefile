.PHONY: install install-gpu train train-ims inference dashboard test clean help

help:           ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:        ## Install project (editable mode)
	pip install -e ".[test]"

install-gpu:    ## Install with CUDA GPU support
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
	pip install -e ".[test]"

train:          ## Train C-MAPSS models (downloads data automatically)
	python scripts/train_all.py

train-ims:      ## Train IMS bearing models
	python scripts/train_ims.py

inference:      ## Run inference and generate recommendations
	python scripts/run_pipeline.py

dashboard:      ## Launch Streamlit dashboard
	streamlit run dashboard/app.py

test:           ## Run all tests
	python -m pytest

clean:          ## Remove caches and generated files
ifeq ($(OS),Windows_NT)
	powershell -Command "Get-ChildItem -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
	powershell -Command "Get-ChildItem -Recurse -Directory -Filter '.pytest_cache' | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
else
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null; true
endif
