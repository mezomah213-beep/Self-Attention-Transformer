import numpy as np
sentence = "What are the symptoms of diabetes?"
tokens = sentence.split()
tokens.append("?")

print("Tokens:", tokens)

d_model = 8
seq_len = len(tokens)
np.random.seed(42)

X = np.random.randn(seq_len, d_model)
W_q = np.random.randn(d_model, d_model)
W_k = np.random.randn(d_model, d_model)
W_v = np.random.randn(d_model, d_model)


Q = X @ W_q

K = X @ W_k

V = X @ W_v

scores = (Q @ K.T) / np.sqrt(d_model)
def softmax(x):

    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))

    return e_x / e_x.sum(axis=-1, keepdims=True)

attention_weights = softmax(scores)

np.set_printoptions(precision=3, suppress=True)
print("\nAttention Weights:")
print(attention_weights)
output = attention_weights @ V
print("\nSelf-Attention Output:")
print(output)