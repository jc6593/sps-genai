from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel
from app.cifar10_classifier import Cifar10Classifier
from app.embedding_model import EmbeddingModel


app = FastAPI(
    title="SPS Generative AI API",
    description=(
        "Generate text, retrieve spaCy embeddings, and classify CIFAR-10 images."
    ),
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
cifar10_classifier = Cifar10Classifier()


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


class ClassificationResponse(BaseModel):
    predicted_class: str
    class_index: int
    confidence: float
    probabilities: dict[str, float]


@app.get("/")
def read_root():
    return {
        "message": "SPS Generative AI API",
        "endpoints": ["/generate", "/embedding", "/classify"],
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


@app.post("/classify", response_model=ClassificationResponse)
async def classify_image(
    image: UploadFile = File(description="An image to classify as CIFAR-10"),
):
    """Classify an uploaded image into one of the ten CIFAR-10 classes."""
    image_bytes = await image.read()

    try:
        with Image.open(BytesIO(image_bytes)) as uploaded_image:
            prepared_image = uploaded_image.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(
            status_code=422,
            detail="The uploaded file is not a readable image.",
        ) from error

    class_index, predicted_class, confidence, probabilities = (
        cifar10_classifier.predict(prepared_image)
    )
    return ClassificationResponse(
        predicted_class=predicted_class,
        class_index=class_index,
        confidence=confidence,
        probabilities=probabilities,
    )
