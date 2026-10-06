import faiss
import numpy as np

vectors = np.array([
    [1.0, 2.0],
    [2.0, 3.0],
    [10.0, 10.0]
], dtype="float32")

index = faiss.IndexFlatL2(2)

index.add(vectors)

print("Number of vectors:", index.ntotal)

question = np.array([
    [1.2, 2.1]
], dtype="float32")

distances, indices = index.search(question, 2)

print("Distances:", distances)
print("Indices:", indices)
