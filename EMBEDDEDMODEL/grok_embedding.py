import tensorflow_hub as hub

# Load the Universal Sentence Encoder (example)
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Use it to generate embeddings
embeddings = model(["Hello, world!", "Sign language recognition"])

