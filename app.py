from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI(title="Batch Sentiment Analysis API", version="1.0.0")

# Public GET/POST API access from browser-based graders.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

analyzer = SentimentIntensityAnalyzer()


class SentimentRequest(BaseModel):
    sentences: List[str] = Field(..., description="Sentences to classify")


class SentimentResult(BaseModel):
    sentence: str
    sentiment: str


class SentimentResponse(BaseModel):
    results: List[SentimentResult]


def classify_sentiment(sentence: str) -> str:
    """Map VADER sentiment scores to the required happy/sad/neutral labels."""
    compound = analyzer.polarity_scores(sentence)["compound"]
    if compound >= 0.05:
        return "happy"
    if compound <= -0.05:
        return "sad"
    return "neutral"


@app.get("/")
def root():
    return {"message": "Batch Sentiment Analysis API is running", "endpoint": "/sentiment"}


@app.post("/sentiment", response_model=SentimentResponse)
def analyze_batch(payload: SentimentRequest):
    return {
        "results": [
            {"sentence": sentence, "sentiment": classify_sentiment(sentence)}
            for sentence in payload.sentences
        ]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000)
