from fastapi import FastAPI  # FastAPI framework import
from fastapi.middleware.cors import CORSMiddleware  # CORS middleware import
from pydantic import BaseModel  # request schema base class

app = FastAPI()  # create app immediately (must be fast)

app.add_middleware(  # add CORS settings
    CORSMiddleware,  # enable CORS
    allow_origins=["*"],  # allow all origins (tighten later)
    allow_credentials=True,  # allow cookies/auth headers
    allow_methods=["*"],  # allow all HTTP methods
    allow_headers=["*"],  # allow all headers
)

class QueryRequest(BaseModel):  # define request body schema
    query: str  # the user query

@app.get("/")  # health check route
def root():  # handler function
    return {"message": "News summarizer is live!"}  # simple response

@app.post("/query")  # main route
def handle_query(req: QueryRequest):  # handler function
    from src.query_and_summarize import query_news  # import lazily to avoid slow startup
    response = query_news(summarizer_model="openai", query=req.query)  # run your pipeline
    return {"summary": response["summary"], "matches": response["matches"]}  # return result
