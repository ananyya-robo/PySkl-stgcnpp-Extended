import pickle
import numpy as np
import sys

result_file = sys.argv[1]

with open(result_file, 'rb') as f:
    scores = np.array(pickle.load(f))

print(f"--- {result_file} ---")
print("Shape:", scores.shape)
print("Row sums (should be ~1.0 if softmax):", scores[:5].sum(axis=1))
print("Min/max values:", scores.min(), scores.max())