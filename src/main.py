from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.query_and_summarize import query_news

app = FastAPI()

# ---------- CORS Middleware ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can later restrict to your GitHub.io URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Request Schema ----------
class QueryRequest(BaseModel):
    query: str

# ---------- Status Check ----------
@app.get("/")
def root():
    return {"message": "News summarizer is live!"}

# ---------- Endpoint ----------
@app.post("/query")
def handle_query(req: QueryRequest):
    response =  query_news(summarizer_model = 'openai', query = req.query)
    return {"summary": response["summary"], "matches": response["matches"]}
