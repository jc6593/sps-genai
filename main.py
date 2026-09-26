from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel


app = FastAPI(
    title="Simple Text Generation and Embedding API",
    description="Generate text with a bigram model and retrieve spaCy embeddings.",
    version="1.0.0",
)


# Training text for the bigram model
corpus = [
    (
        "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
        "It tells the story of Edmond Dantès, who is falsely imprisoned "
        "and later seeks revenge."
    ),
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]


# Build the model once when the application starts
bigram_model = BigramModel(corpus)
embedding_model = EmbeddingModel()


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int = Field(ge=1, le=100)


class EmbeddingRequest(BaseModel):
    query_word: str = Field(
        min_length=1,
        description="A word to embed, for example 'apple'.",
    )


class EmbeddingResponse(BaseModel):
    query_word: str
    dimension: int
    embedding: list[float]


@app.get("/")
def read_root():
    return {
        "message": "Simple Text Generation and Embedding API",
        "endpoints": ["/generate", "/embedding"],
    }


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(
        request.start_word,
        request.length,
    )

    return {"generated_text": generated_text}


@app.post("/embedding", response_model=EmbeddingResponse)
def create_embedding(request: EmbeddingRequest):
    """Return the spaCy embedding for the requested word."""
    normalized_word = request.query_word.strip()

    try:
        embedding = embedding_model.get_embedding(normalized_word)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return EmbeddingResponse(
        query_word=normalized_word,
        dimension=len(embedding),
        embedding=embedding,
    )
