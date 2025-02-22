ALL_PACKAGES := src tests

.PHONY: reformat lint

lint:
	poetry run ruff check $(ALL_PACKAGES) &
	poetry run mypy .

reformat:
	poetry run ruff format $(ALL_PACKAGES)

api:
	poetry run python -m uvicorn src.vector_retrieval_service.service_api.fastapi_app:app --reload

#install_torch:
#	poetry source add -p explicit pytorch https://download.pytorch.org/whl/cu121
