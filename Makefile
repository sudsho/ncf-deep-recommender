.PHONY: install train eval serve test docker clean

install:
	pip install -r requirements.txt

train:
	python -m src.train --config configs/default.yaml

eval:
	python -m src.evaluate --config configs/default.yaml

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -q tests/

docker:
	docker build -t ncf-deep-recommender:latest .

clean:
	rm -rf __pycache__ .pytest_cache *.egg-info build dist
	find . -name "*.pyc" -delete
