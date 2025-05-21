import numpy as np
import torch
from sentence_transformers import SentenceTransformer

print("✅ NumPy version:", np.__version__)
print("✅ Torch version:", torch.__version__)

try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(["test sentence"])
    print("✅ Embedding shape:", embedding.shape)
except Exception as e:
    print("❌ Error during embedding:", e)
