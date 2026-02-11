import tensorflow_hub as hub

# Load the Universal Sentence Encoder (cached after first download)
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Define your input text
texts = [
    "Sign language recognition is fascinating.",
    "Neural networks help interpret gestures."
]

# Generate embeddings
embeddings = model(texts)

# Print the embeddings
for i, emb in enumerate(embeddings):
    print(f"Embedding for: '{texts[i]}':\n{emb.numpy()}\n")
