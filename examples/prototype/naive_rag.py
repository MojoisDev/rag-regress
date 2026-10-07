import faiss
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

# -----------------------------
# 1. Chunking
# -----------------------------

def create_chunks(text, chunk_size=50, overlap=10):
    words = text.split()

    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# -----------------------------
# 2. Our document
# -----------------------------

document = """
Computer networks allow multiple devices to communicate and share resources.
TCP is a connection-oriented protocol that provides reliable communication.
TCP uses acknowledgements, sequencing and retransmission to ensure that data
reaches its destination correctly. The TCP three-way handshake establishes
a connection using SYN, SYN-ACK and ACK messages. UDP is a connectionless
protocol that is often used when speed is more important than reliability.
DNS translates domain names into IP addresses. HTTP is used for communication
between web browsers and web servers.
"""


# -----------------------------
# 3. Create chunks
# -----------------------------

chunks = create_chunks(
    document,
    chunk_size=25,
    overlap=5
)

print("Created", len(chunks), "chunks.")


# -----------------------------
# 4. Load embedding model
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# 5. Create embeddings
# -----------------------------

chunk_embeddings = model.encode(chunks)


# -----------------------------
# 6. User question
# -----------------------------

question = "What is TCP?"
question_embedding = model.encode([question])[0]


# -----------------------------
# 7. Cosine similarity
# -----------------------------

def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# -----------------------------
# 8. Find similarities
# -----------------------------

index = faiss.IndexFlatL2(384)

index.add(np.array(chunk_embeddings, dtype="float32"))

top_k = 2

distances, indices = index.search(
    np.array([question_embedding], dtype="float32"),
    top_k
)

top_results = []

for distance, index_number in zip(distances[0], indices[0], strict=True):
    chunk = chunks[index_number]
    top_results.append((chunk, distance))

print("\nRetrieved context:\n")

for chunk, similarity in top_results:
    print(f"Distance: {similarity:.4f}")
    print(chunk)
    print()


# -----------------------------
# 11. Build context
# -----------------------------

if top_results:
    context = "\n\n".join(
        chunk for chunk, similarity in top_results
    )
else:
    context = "No relevant information was found in the document."


# -----------------------------
# 12. Build prompt
# -----------------------------

prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context,
say that the information is not available in the provided context.

Context:
{context}

Question:
{question}

Answer:
"""


# -----------------------------
# 13. Send to local LLM
# -----------------------------

response = ollama.chat(
    model="qwen2.5:1.5b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# -----------------------------
# 14. Display answer
# -----------------------------

answer = response["message"]["content"]

print("========== ANSWER ==========\n")
print(answer)
