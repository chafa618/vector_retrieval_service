from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from vector_retrieval_service.embedding_retriever.models.model_factory import LLMFactory
from vector_retrieval_service.service_api.fastapi_views import retrival_service



app = FastAPI()
app.include_router(retrival_service)

@app.get("/")
async def home() -> dict[str, str]:
    return {"message": "Hello!"}

@app.get("/healthcheck")
def healthcheck() -> dict[str, str]:
    return {"message": "App is running!"}


@app.get("/")
def redirect_to_docs() -> RedirectResponse:
    return RedirectResponse(url="/docs", status_code=302)


"""@app.on_event("startup")
def startup_event():
    print("App started")
    model = LLMFactory.get_model("mini_lm")
    model.get_embedding("Startup!")"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)