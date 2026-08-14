"""
Tiny GPT From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_vocab
def build_vocab(text):
    """Return a sorted list of unique characters in text."""
    # TODO: return a sorted list of every unique character in text
    char_set = set()
    for char in text:
        char_set.add(char)
    
    result = sorted(char_set)

    return result

# Step 2 - build_stoi
def build_stoi(vocab):
    """Return a dict mapping each character in vocab to its index."""
    # TODO: map each character in vocab to its integer position
    dict = {}
    for idx, char in enumerate(vocab):
        dict[char] = idx

    return dict

# Step 3 - build_itos
def build_itos(vocab):
    """Return a dict mapping each index 0..len(vocab)-1 to its character."""
    # TODO: build an int-to-string lookup from the vocab list
    dict = {}
    for idx, char in enumerate(vocab):
        dict[idx] = char
    
    return dict

# Step 4 - encode_char
def encode_char(ch, stoi):
    """Return the integer token id for a single character ch using stoi."""
    # TODO: look up ch in the stoi mapping and return its id
    return stoi[ch]

# Step 5 - encode_string
def encode_string(text, stoi):
    """Encode a full string into a list of token ids using stoi."""
    # TODO: map each char in text through stoi (via encode_char) into a list of ids
    res = []
    for char in text:
        res.append(stoi[char])
    
    return res

# Step 6 - decode_int
def decode_int(token_id, itos):
    """Return the single character mapped to token_id by itos."""
    return itos[token_id]

# Step 7 - decode_ids
def decode_ids(ids, itos):
    """Decode a list of token ids into a string using itos."""
    res = []
    for id in ids:
        res.append(itos[id])
    
    return ''.join(res)

# Step 8 - make_1d_array
import numpy as np

def make_1d_array(values):
    """Create a 1D NumPy array from a Python list of numbers."""
    return np.array(values)

# Step 9 - get_array_shape
import numpy as np

def get_array_shape(arr):
    """Return the shape tuple of a NumPy array."""
    return arr.shape

# Step 10 - get_array_dtype
import numpy as np

def get_array_dtype(arr):
    """Return the dtype of a NumPy array."""
    return arr.dtype

# Step 11 - make_2d_zeros
import numpy as np

def make_2d_zeros(rows, cols):
    """Return a 2D NumPy array of zeros with shape (rows, cols)."""
    return np.zeros((rows, cols))

# Step 12 - make_2d_random
import numpy as np

def make_2d_random(rows, cols, seed):
    """Return a (rows, cols) array of uniform floats in [0, 1) seeded by `seed`."""
    rng = np.random.default_rng(seed)
    return rng.random((rows, cols))

# Step 13 - index_element
def index_element(arr, i, j):
    """Return the scalar element at position (i, j) of a 2D array."""
    return arr[i, j]

# Step 14 - slice_row
import numpy as np

def slice_row(arr, i):
    """Return row i of a 2D array as a 1D view."""
    return arr[i, :]

# Step 15 - slice_column
import numpy as np

def slice_column(arr, j):
    """Return column j of a 2D array as a 1D array of length R."""
    # TODO: index into arr to extract the j-th column as a 1D array.
    return arr[:, j]

# Step 16 - slice_subblock
import numpy as np

def slice_subblock(arr, r0, r1, c0, c1):
    """Return the sub-block arr[r0:r1, c0:c1] of a 2D array."""
    return arr[r0:r1, c0:c1]

# Step 17 - elementwise_add
import numpy as np

def elementwise_add(a, b):
    """Return the elementwise sum of two same-shape arrays."""
    return a + b

# Step 18 - elementwise_multiply
import numpy as np

def elementwise_multiply(a, b):
    """Return the elementwise product of two same-shape arrays."""
    # TODO: compute the elementwise (Hadamard) product of a and b
    return a * b

# Step 19 - scalar_broadcast_add
import numpy as np

def scalar_broadcast_add(arr, scalar):
    """Return a new array equal to arr with scalar added to every element."""
    return arr + scalar

# Step 20 - vector_matrix_broadcast_add
import numpy as np

def vector_matrix_broadcast_add(matrix, vector):
    """Add a 1D vector to each row of a 2D matrix via broadcasting."""
    return matrix + vector

# Step 21 - array_exp
import numpy as np

def array_exp(arr):
    """Return the elementwise exponential of arr."""
    return np.exp(arr)

# Step 22 - array_log
import numpy as np

def array_log(arr):
    """Return the elementwise natural log of arr (assumes arr > 0)."""
    # TODO: apply elementwise natural log to arr and return the result
    return np.log(arr)

# Step 23 - sum_all
import numpy as np

def sum_all(arr):
    """Return the sum of every element of arr as a scalar."""
    return np.sum(arr)

# Step 24 - sum_axis0
import numpy as np

def sum_axis0(arr):
    """Sum a 2D array along axis 0, collapsing rows into a 1D vector of column sums."""
    return np.sum(arr, axis=0)

# Step 25 - sum_axis1
import numpy as np

def sum_axis1(arr):
    """Sum a 2D array along axis 1, returning a 1D array of row sums."""
    return np.sum(arr, axis=1)

# Step 26 - max_along_axis
import numpy as np

def max_along_axis(arr, axis):
    """Return the maximum of arr along the given axis, with that axis removed."""
    return np.max(arr, axis)

# Step 27 - matmul
import numpy as np

def matmul(a, b):
    """Return the matrix product a @ b for 2D arrays a (M,K) and b (K,N)."""
    return a @ b

# Step 28 - transpose_matrix
def transpose_matrix(arr):
    """Return the transpose of a 2D array."""
    return arr.T

# Step 29 - sum_keepdims
import numpy as np

def sum_keepdims(arr, axis):
    """Sum along `axis` while keeping that dimension as size 1."""
    return np.sum(arr, axis=axis, keepdims=True)

# Step 30 - naive_softmax_1d
import numpy as np

def naive_softmax_1d(logits):
    """Compute softmax of a 1D logits vector via the direct exp/sum formula."""
    return np.exp(logits) / np.sum(np.exp(logits))

# Step 31 - softmax_overflow_demo
def softmax_overflow_demo(large_value):
    """Show that naive exp overflows on a large logit.

    Return {'naive_exp': float, 'overflowed': bool}.
    """
    naive_exp = float(np.exp(large_value))
    return {'naive_exp': naive_exp, 'overflowed': np.isinf(naive_exp)}

# Step 32 - stable_softmax_1d
import numpy as np

def stable_softmax_1d(logits):
    """Numerically stable softmax over a 1D logits vector."""
    shifted = logits - np.max(logits)
    return np.exp(shifted) / np.sum(np.exp(shifted))

# Step 33 - stable_softmax_2d_rowwise
import numpy as np

def stable_softmax_2d_rowwise(logits):
    """Row-wise numerically stable softmax of a 2D logits array."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)

# Step 34 - read_text_file
def read_text_file(text_blob):
    """Return text_blob unchanged after validating it is a non-empty string."""
    # TODO: validate that text_blob is a non-empty str and return it as the corpus string
    if len(text_blob) == 0:
        raise ValueError
    if not isinstance(text_blob, str):
        raise TypeError
    
    return text_blob

# Step 35 - encode_corpus_to_int_array
def encode_corpus_to_int_array(text, stoi):
    """Convert the corpus string into a 1D NumPy int64 array of token ids."""
    def encode_string(text, stoi):
        """Encode a full string into a list of token ids using stoi. Unknown characters map to UNK_ID."""
        return [stoi.get(ch) for ch in text]

    encoded_text = encode_string(text, stoi)
    return np.array(encoded_text)

# Step 36 - pick_split_point
import math
def pick_split_point(n, train_frac):
    """Return integer split index so data[:idx] is train and data[idx:] is val."""
    # TODO: compute the integer split index from n and train_frac
    return n * train_frac if isinstance(n * train_frac, int) else math.floor(n * train_frac)

# Step 37 - slice_train_and_val
def slice_train_and_val(data, split_idx):
    """Split a 1D token-id array into (train, val) at split_idx."""
    # TODO: return (data[:split_idx], data[split_idx:])
    return (data[:split_idx], data[split_idx:])

# Step 38 - pick_block_size
def pick_block_size(default_size):
    """Return the context length (block_size) for training windows."""
    return default_size if default_size > 1 else 1

# Step 39 - slice_x_at_offset
import numpy as np

def slice_x_at_offset(data, i, block_size):
    """Return the input window data[i : i + block_size]."""
    return data[i : i + block_size]

# Step 40 - slice_y_at_offset
import numpy as np

def slice_y_at_offset(data, i, block_size):
    """Return the target window of length block_size starting at i+1."""
    # TODO: extract the target window Y = data[i+1 : i+1+block_size] shifted by one.
    return data[i+1 : i+1+block_size]

# Step 41 - sample_random_batch_offsets
def sample_random_batch_offsets(data_len, block_size, batch_size, rng):
    """Sample batch_size random valid starting offsets for (block_size+1)-windows."""
    max_offset = data_len - block_size - 1
    return rng.integers(0, max_offset, size=batch_size)

# Step 42 - stack_x_batch
import numpy as np

def stack_x_batch(data, offsets, block_size):
    """Stack per-offset X windows into a 2D batch matrix of shape (B, block_size)."""
    # TODO: for each offset, take a length-block_size slice of data and stack them as rows
    def slice_x_at_offset(data, i, block_size):
        """Return the input window data[i : i + block_size]."""
        return data[i : i + block_size]

    return np.array([slice_x_at_offset(data, offset, block_size) for offset in offsets])

# Step 43 - stack_y_batch
import numpy as np

def stack_y_batch(data, offsets, block_size):
    """Stack per-offset Y windows into a 2D (B, block_size) target matrix."""
    # TODO: for each offset, take the length-block_size slice starting at i+1 and stack rows
    def slice_y_at_offset(data, i, block_size):
        """Return the target window of length block_size starting at i+1."""
        return data[i+1 : i+1+block_size]

    return np.array([slice_y_at_offset(data, offset, block_size) for offset in offsets])

# Step 44 - get_batch
def get_batch(data, block_size, batch_size, rng):
    # TODO: package one training batch (X, Y) of shape (batch_size, block_size) from data using rng.
    def sample_random_batch_offsets(data_len, block_size, batch_size, rng):
        """Sample batch_size random valid starting offsets for (block_size+1)-windows."""
        max_offset = data_len - block_size
        return rng.integers(0, max_offset, size=batch_size)

    def slice_x_at_offset(data, i, block_size):
        """Return the input window data[i : i + block_size]."""
        return data[i : i + block_size]

    def slice_y_at_offset(data, i, block_size):
        """Return the target window of length block_size starting at i+1."""
        return data[i+1 : i+1+block_size]
    
    data_len = len(data)
    offsets = sample_random_batch_offsets(data_len, block_size, batch_size, rng)
    X = np.array([slice_x_at_offset(data, i, block_size) for i in offsets])
    Y = np.array([slice_y_at_offset(data, i, block_size) for i in offsets])
    return X, Y

# Step 45 - allocate_count_matrix
import numpy as np

def allocate_count_matrix(vocab_size):
    """Allocate a (V, V) integer zero matrix for bigram counts."""
    return np.zeros((vocab_size, vocab_size), dtype=np.int64)

# Step 46 - loop_fill_counts
import numpy as np

def loop_fill_counts(n_matrix, data):
    """Increment n_matrix[curr, next] for every consecutive pair in data."""
    for t in range(len(data) - 1):
        i = data[t]
        j = data[t + 1]
        n_matrix[i][j] += 1
    
    return n_matrix

# Step 47 - vectorize_counts_add_at
import numpy as np

def vectorize_counts_add_at(vocab_size, data):
    """Build (V, V) bigram counts from a 1D id array using vectorized scatter-add."""
    # TODO: allocate counts, then scatter-add 1 at each (data[:-1], data[1:]) pair
    def allocate_count_matrix(vocab_size):
        """Allocate a (V, V) integer zero matrix for bigram counts."""
        return np.zeros((vocab_size, vocab_size), dtype=np.int64)
    
    matrix = allocate_count_matrix(vocab_size)
    cur = data[:-1]
    next_ = data[1:]
    np.add.at(matrix, (cur, next_), 1)
    
    return matrix

# Step 48 - add_one_smoothing
import numpy as np

def add_one_smoothing(n_matrix):
    """Return n_matrix with every entry incremented by 1 (Laplace smoothing)."""
    # TODO: apply +1 Laplace smoothing to the bigram count matrix
    return n_matrix + 1

# Step 49 - row_sums_of_counts
def row_sums_of_counts(n_matrix):
    """Return per-row sums of n_matrix with shape (V, 1)."""
    def sum_keepdims(arr, axis):
        """Sum along `axis` while keeping that dimension as size 1."""
        return np.sum(arr, axis=axis, keepdims=True)
    
    return sum_keepdims(n_matrix, 1)

# Step 50 - normalize_counts_to_probs
def normalize_counts_to_probs(n_matrix):
    """Normalize a (V, V) count matrix into a row-stochastic probability matrix."""
    def row_sums_of_counts(n_matrix):
        """Return per-row sums of n_matrix with shape (V, 1)."""
        def sum_keepdims(arr, axis):
            """Sum along `axis` while keeping that dimension as size 1."""
            return np.sum(arr, axis=axis, keepdims=True)
    
        return sum_keepdims(n_matrix, 1)
    return n_matrix / row_sums_of_counts(n_matrix)

# Step 51 - sample_next_token
def sample_next_token(p_matrix, current_id, rng):
    """Sample the next token id from P[current_id] using rng."""
    probs = p_matrix[current_id]
    V = len(probs)
    return int(rng.choice(V, p=probs))

# Step 52 - generate_sequence
def generate_sequence(p_matrix, start_id, length, rng):
    """Autoregressively sample `length` token ids from a bigram matrix, starting with `start_id`."""
    # TODO: build a length-L int array starting at start_id, then sample each next id from p_matrix
    def sample_next_token(p_matrix, current_id, rng):
        """Sample the next token id from P[current_id] using rng."""
        probs = p_matrix[current_id]
        V = len(probs)
        return int(rng.choice(V, p=probs))
    
    tokens = [start_id]
    for _ in range(length - 1):
        next_token = sample_next_token(p_matrix, tokens[-1], rng)
        tokens.append(next_token)
    
    return np.array(tokens)

# Step 53 - decode_generated_sequence
def decode_generated_sequence(ids, itos):
    """Decode a generated 1D array/list of token ids into a string via itos."""
    # TODO: turn ids into a readable string using itos
    decoded_string = []
    for key in ids:
        char = itos[int(key)]
        decoded_string.append(char)
    
    return "".join(decoded_string)

# Step 54 - log_prob_of_pair
def log_prob_of_pair(p_matrix, current_id, next_id):
    """Return the log probability of a single (current, next) bigram."""
    # TODO: pick out P[current_id, next_id] and return its natural log
    def index_element(arr, i, j):
        """Return the scalar element at position (i, j) of a 2D array."""
        return arr[i, j]
    
    def array_log(arr):
        """Return the elementwise natural log of arr (assumes arr > 0)."""
        # TODO: apply elementwise natural log to arr and return the result
        return np.log(arr)
    
    return array_log(p_matrix[current_id, next_id])

# Step 55 - sum_negative_log_probs
def sum_negative_log_probs(p_matrix, data):
    # TODO: sum the negative log probabilities of all consecutive bigrams in data
    total = 0.0
    for t in range(len(data) - 1):
        total += log_prob_of_pair(p_matrix, data[t], data[t+1])
    return -float(total)

# Step 56 - average_nll
def average_nll(p_matrix, data):
    total = sum_negative_log_probs(p_matrix, data)
    return total / (len(data) - 1)

# Step 57 - initialize_w_random
import numpy as np

def initialize_w_random(vocab_size, rng):
    """Return a (vocab_size, vocab_size) float64 matrix of N(0,1) samples drawn from rng."""
    # TODO: sample a (vocab_size, vocab_size) array of standard normal values using rng
    return rng.standard_normal(size=(vocab_size, vocab_size), dtype=np.float64)

# Step 58 - scale_w_small
import numpy as np

def scale_w_small(w_matrix, scale):
    """Return w_matrix scaled by the given small factor."""
    # TODO: return a new array equal to w_matrix multiplied by scale
    return w_matrix * scale

# Step 59 - one_hot_encode_batch
import numpy as np

def one_hot_encode_batch(ids, vocab_size):
    """Convert a 1D array of token ids into a (N, vocab_size) one-hot matrix."""
    matrix = make_2d_zeros(len(ids), vocab_size)
    for idx, key in enumerate(ids):
        matrix[idx, key] = 1

    return matrix

# Step 60 - forward_logits_onehot
def forward_logits_onehot(onehot, w_matrix):
    return onehot @ w_matrix

# Step 61 - observe_lookup_equivalence
import numpy as np

def observe_lookup_equivalence(w, ids):
    """Show that one-hot @ W equals W[ids] for a small example.
    Returns a dict with keys 'onehot_result' and 'index_result'.
    """
    one_hot = one_hot_encode_batch(ids, w.shape[0])
    onehot_result = one_hot @ w
    index_result = w[ids]
    return {'onehot_result': onehot_result, 'index_result': index_result}

# Step 62 - forward_logits_lookup
def forward_logits_lookup(w, ids):
    """Return logits (B, V) by gathering rows of w at positions ids."""
    return w[ids]

# Step 63 - logits_to_probs_rowwise
def logits_to_probs_rowwise(logits):
    # TODO: convert a (B, V) logits matrix into a row-wise probability matrix
    def stable_softmax_2d_rowwise(logits):
        """Row-wise numerically stable softmax of a 2D logits array."""
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)
    return stable_softmax_2d_rowwise(logits)

# Step 64 - gather_correct_token_probs
def gather_correct_token_probs(probs, targets):
    """Return probs[i, targets[i]] for each i, shape (B,)."""
    return probs[np.arange(len(targets)), targets]

# Step 65 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, targets):
    """Mean negative log-likelihood over a batch."""
    probs = gather_correct_token_probs(probs, targets)
    log_probs = array_log(probs)
    return (-1 / len(targets)) * log_probs.sum()

# Step 66 - derive_dlogits_on_paper
def derive_dlogits_on_paper():
    """Return a string summarizing the derivation of dL/dlogits for mean cross-entropy."""

    return (
        "Derivation of dL/dlogits:\n\n"
        "1. Loss: L = -1/B * sum(log(probs[i, targets[i]]))\n\n"
        "2. Softmax: probs[i,j] = exp(logits[i,j]) / sum(exp(logits[i,k]))\n\n"
        "3. For the correct class j == targets[i]:\n"
        "   dL/dlogits[i,j] = (probs[i,j] - 1) / B\n\n"
        "4. For incorrect class j != targets[i]:\n"
        "   dL/dlogits[i,j] = probs[i,j] / B\n\n"
        "5. Combined final formula:\n"
        "   dL/dlogits = (probs - onehot(targets)) / B"
    )

# Step 67 - compute_dlogits
def compute_dlogits(probs, targets):
    """Gradient of mean cross-entropy w.r.t. logits. probs: (B,V), targets: (B,)."""
    B, V = probs.shape
    onehot = one_hot_encode_batch(targets, V)
    return (probs - onehot) / B

# Step 68 - derive_dw_on_paper
def derive_dw_on_paper():
    """Return a short written derivation of dL/dW for the lookup-as-matmul forward."""
    return (
        "Forward: logits = onehot(ids) @ W, equivalently logits[b] = W[ids[b]].\n"
        "Shapes: ids (B,), onehot O (B, V), W (V, D), logits (B, D), dlogits (B, D).\n"
        "Chain rule: dL/dW = O.T @ dlogits, shape (V, D).\n"
        "Since O has a single 1 per row at column ids[b], O.T @ dlogits sums rows of dlogits into rows of dW.\n"
        "Row v of dW equals the sum of dlogits[b] over all b with ids[b] == v.\n"
        "Implementation: scatter-add dlogits rows into dW at indices ids."
    )

# Step 69 - compute_dw_scatter_add
import numpy as np

def compute_dw_scatter_add(ids, dlogits, vocab_size):
    """Scatter-add dlogits rows into dW at positions given by ids."""
    dW = np.zeros((vocab_size, dlogits.shape[1]))
    np.add.at(dW, ids, dlogits)
    return dW

# Step 70 - sgd_update_w
import numpy as np

def sgd_update_w(w, dw, learning_rate):
    """Apply one SGD step: return w - learning_rate * dw as a new array."""
    return w - dw * learning_rate

# Step 71 - run_one_training_step
def run_one_training_step(w, ids, targets, learning_rate):
    """Run forward, loss, backward, and SGD update once. Return {'w': new_w, 'loss': float}."""
    #forward
    forward_logits = w[ids]
    probs = stable_softmax_2d_rowwise(forward_logits)
    # loss
    loss = cross_entropy_loss(probs, targets)
    # backward
    B, V = probs.shape
    dlogits = compute_dlogits(probs, targets)
    dw = compute_dw_scatter_add(ids, dlogits, w.shape[0])
    # update
    new_w = sgd_update_w(w, dw, learning_rate)
    return {'w': new_w, 'loss': float(loss)}

# Step 72 - train_neural_bigram_loop
def train_neural_bigram_loop(w, data, block_size, batch_size, learning_rate, num_steps, log_every):
    """Run the neural bigram training loop and return {'w', 'loss_history'}."""
    loss_history = []
    current_w = w
    rng = np.random.default_rng(0)

    for step in range(num_steps):
        X, Y = get_batch(data, block_size, batch_size, rng)
        ids = X.flatten()
        targets = Y.flatten()
        result = run_one_training_step(current_w, ids, targets, learning_rate)

        current_w = result['w']

        if step % log_every == 0:
            loss_history.append(result['loss'])
           
    
    return {'w': current_w, 'loss_history': loss_history}

# Step 73 - sample_from_neural_bigram
def sample_from_neural_bigram(w, start_id, num_tokens, itos):
    """Generate a string by repeatedly sampling from softmax of W[id]."""
    # TODO: starting from start_id, sample num_tokens new ids and decode the full sequence...
    ids = [start_id]
    current_id = start_id
    rng = np.random.default_rng(0)
    for _ in range(num_tokens):
        logits = forward_logits_lookup(w, np.array([current_id]))
        probs = logits_to_probs_rowwise(logits)
        next_token = int(rng.choice(len(itos), p=probs[0]))
        ids.append(next_token)
        current_id = next_token
    
    sequence = decode_ids(ids, itos)
    return sequence

# Step 74 - linear_forward
def linear_forward(x, w):
    Y = x @ w
    return {'y': Y, 'cache': {'x': x, 'w': w}}

# Step 75 - derive_dx_on_paper
def derive_dx_on_paper():
    """Return notes deriving dL/dX = dY @ W.T for Y = X @ W."""
    return ('Y = X @ W\ndL/dX = dY @ W.T\nshapes: X (B, In), W (In, Out), dY (B, Out) -> dL/dX (B, In)')

# Step 76 - derive_linear_dw_on_paper
def derive_linear_dw_on_paper():
    """Return a string with the derivation of dL/dW for Y = X @ W."""
    return(
        "Derivation of dL/dW for Y = X @ W:\n\n"
        "Forward: Y = X @ W, shapes X (B, D_in), W (D_in, D_out), Y (B, D_out).\n\n"
        "Upstream gradient dY = dL/dY, shape (B, D_out), is given.\n\n"
        "Chain rule: dL/dW[k,j] = sum_b dL/dY[b,j] * dY[b,j]/dW[k,j].\n\n"
        "Since Y[b,j] = sum_k X[b,k] * W[k,j], we have dY[b,j]/dW[k,j] = X[b,k].\n\n"
        "So dL/dW[k,j] = sum_b X[b,k] * dY[b,j] = (X.T @ dY)[k,j].\n\n"
        "Combined final formula: dL/dW = X.T @ dY, shape (D_in, D_out)."
    )

# Step 77 - linear_backward_dx
def linear_backward_dx(dy, cache):
    # TODO: compute the gradient of the loss w.r.t. the linear layer input X given dy and cache
    w = cache['w']
    return dy @ w.T

# Step 78 - linear_backward_dw
def linear_backward_dw(dy, cache):
    """Return dL/dW for a linear layer Y = X @ W."""
    # TODO: compute the weight gradient using x from cache and the upstream dy
    X = cache['x']
    return X.T @ dy

# Step 79 - bias_add_forward
def bias_add_forward(x, b):
    """Add bias vector b (D,) to every row of x (B, D).

    Returns {'y': ndarray (B, D), 'cache': {'b_shape': tuple}}.
    """
    y = vector_matrix_broadcast_add(x, b)
    return {'y': y, 'cache': {'b_shape': b.shape}}

# Step 80 - bias_add_backward_db
def bias_add_backward_db(dy, cache):
    """Compute db from upstream gradient dy for y = x + b."""
    return dy.sum(axis=0)

# Step 81 - relu_forward
def relu_forward(x):
    """Apply elementwise ReLU and cache the input for backward.

    Returns a dict with keys 'y' (activated array) and 'cache' (dict with 'x').
    """
    # TODO: apply elementwise ReLU and cache the input for backward.
    y = np.maximum(0, x)
    return {'y': y, 'cache': {'x': x}}

# Step 82 - relu_backward
def relu_backward(dy, cache):
    """Backward pass for ReLU. cache['x'] holds the original input."""
    x = cache['x']
    return dy * (x > 0)

# Step 83 - softmax_cross_entropy_backward
def softmax_cross_entropy_backward(probs, targets):
    """Return dL/dlogits for mean cross-entropy with softmax probs."""
    # TODO: produce the (B, V) gradient of mean cross-entropy w.r.t. logits.
    B, V = probs.shape
    onehot = one_hot_encode_batch(targets, V)
    return (probs - onehot)/B

# Step 84 - layernorm_forward_mean
import numpy as np

def layernorm_forward_mean(x):
    """Return the per-row mean of x with shape (B, 1)."""
    # TODO: compute the per-row mean of x, preserving the reduced axis as size 1
    D = x.shape[-1]
    return sum_keepdims(x, -1) / D

# Step 85 - layernorm_forward_variance
import numpy as np

def layernorm_forward_variance(x, mean):
    """Compute the per-row (biased) variance of x given its per-row mean.

    Args:
        x: ndarray of shape (B, D).
        mean: ndarray of shape (B, 1), the per-row mean of x.

    Returns:
        var: ndarray of shape (B, 1), the per-row variance.
    """
    return sum_keepdims((x - mean)**2, -1) / x.shape[-1]

# Step 86 - layernorm_forward_normalize
import numpy as np

def layernorm_forward_normalize(x, mean, var, eps):
    """Normalize each row of x to zero mean and unit variance."""
    # TODO: subtract the per-row mean and divide by sqrt(var + eps)
    return (x - mean) / np.sqrt(var + eps)

# Step 87 - layernorm_forward_affine
def layernorm_forward_affine(x, gamma, beta, eps):
    """Run LayerNorm forward over rows of x with affine params gamma, beta."""
    # TODO: normalize each row to zero mean / unit variance, then apply gamma and beta.
    mean = layernorm_forward_mean(x)
    variance = layernorm_forward_variance(x, mean)
    x_hat = layernorm_forward_normalize(x, mean, variance, eps)
    y = x_hat * gamma + beta
    return {'y': y, 'cache': {'x':x, 'x_hat': x_hat, 'mean': mean, 'var':variance, 'gamma':gamma, 'eps': eps}}

# Step 88 - layernorm_backward_subtract_mean
import numpy as np

def layernorm_backward_subtract_mean(dy, cache):
    """Gradient through y = x - mean(x, axis=1, keepdims=True).

    dy: (B, D) upstream gradient w.r.t. the centered output.
    cache: dict with keys 'x' (B, D) and 'mean' (B,).
    Returns dx of shape (B, D).
    """
    D = dy.shape[1]
    dy_mean = sum_keepdims(dy, 1) / D
    return dy - dy_mean

# Step 89 - layernorm_backward_divide_std
def layernorm_backward_divide_std(dy, cache):
    """Propagate dy through the divide-by-std step of LayerNorm."""
    # TODO: propagate the upstream gradient through the divide-by-std step of LayerNorm
    var = cache['var']
    eps = cache['eps']
    return dy / np.sqrt(var + eps)

# Step 90 - layernorm_backward_full
import numpy as np

def layernorm_backward_full(dy, cache):
    """Full LayerNorm backward. Return {'dx', 'dgamma', 'dbeta'}."""
    x = cache['x']
    x_hat = cache['x_hat']
    var = cache['var']
    eps = cache['eps']
    gamma = cache['gamma']
    std = np.sqrt(var + eps)
    D = x.shape[1]
    
    dgamma = np.sum(dy * x_hat, axis=0)
    dbeta = np.sum(dy, axis=0)
    dx_hat = dy * gamma
    
    dx = (1.0 / std) * (
        dx_hat 
        - np.mean(dx_hat, axis=1, keepdims=True) 
        - x_hat * np.mean(dx_hat * x_hat, axis=1, keepdims=True)
    )
    
    return {'dx': dx, 'dgamma': dgamma, 'dbeta': dbeta}

# Step 91 - layernorm_backward_implementation
def layernorm_backward_implementation(d_out, cache):
    # TODO: return {'dx', 'dgamma', 'dbeta'} gradients for LayerNorm given d_out and the forward cache.
    x = cache['x']
    mean = cache['mean']
    var = cache['var']
    x_hat = cache['x_hat']
    gamma = cache['gamma']
    beta = cache['beta']
    eps = cache['eps']
    std = np.sqrt(var + eps)
    D = x.shape[1]
    
    dgamma = np.sum(d_out * x_hat, axis=0)
    dbeta = np.sum(d_out, axis=0)
    dx_hat = d_out * gamma
    
    dx = (1.0 / std) * (
        dx_hat
        - np.mean(dx_hat, axis=1, keepdims=True)
        - x_hat * np.mean(dx_hat * x_hat, axis=1, keepdims=True)
    )
    
    return {'dx': dx, 'dgamma': dgamma, 'dbeta': dbeta}

# Step 92 - create_token_embedding
def create_token_embedding(vocab_size, d_model, scale=0.02):
    """Initialize the token embedding matrix E of shape (vocab_size, d_model)."""
    np.random.seed(0)
    return np.random.standard_normal((vocab_size, d_model))*scale

# Step 93 - token_embedding_forward
def token_embedding_forward(token_ids, embedding_matrix):
    """Look up token embeddings for a batch of integer token ids.

    Inputs:
        token_ids: ndarray of shape (B, T), dtype int
        embedding_matrix: ndarray of shape (V, d_model)
    Returns:
        out: ndarray of shape (B, T, d_model)
        cache: dict with keys 'token_ids', 'vocab_size'
    """
    # TODO: look up the embedding row for each token id and build the cache
    vocab_size = embedding_matrix.shape[0]
    cache = {'token_ids': token_ids, 'vocab_size': vocab_size}
    return embedding_matrix[token_ids], cache

# Step 94 - token_embedding_backward
import numpy as np

def token_embedding_backward(d_out, cache):
    # TODO: scatter-add d_out into a (vocab_size, d_model) dE using cache['token_ids'].
    ids = cache['token_ids']
    vocab_size = cache['vocab_size']
    d_model = d_out.shape[-1]
    dE = np.zeros((vocab_size, d_model))
    np.add.at(dE, ids, d_out)
    return dE

# Step 95 - create_positional_embedding
def create_positional_embedding(block_size, d_model, scale=0.02):
    """Initialize the learned positional embedding matrix P of shape (block_size, d_model)."""
    # TODO: build a (block_size, d_model) matrix of small random values scaled by `scale`
    P = make_2d_random(block_size, d_model, seed=None)
    return scale_w_small(P, scale)

# Step 96 - slice_positional_embedding
import numpy as np

def slice_positional_embedding(positional_matrix, seq_len):
    """Return the first seq_len rows of the positional embedding matrix."""
    # TODO: return the leading seq_len rows of positional_matrix as a (seq_len, d_model) array.
    return positional_matrix[:seq_len]

# Step 97 - add_token_and_positional_embeddings
def add_token_and_positional_embeddings(token_emb, pos_emb):
    """Sum token embeddings (B,T,d_model) and positional embeddings (T,d_model)."""
    # TODO: combine token and positional embeddings into a single (B,T,d_model) tensor
    return token_emb + pos_emb

# Step 98 - embedding_sum_backward
def embedding_sum_backward(d_out):
    """Backprop through H = token_emb + pos_emb (with broadcasting over batch)."""
    d_token_emb = d_out
    d_pos_emb = sum_axis0(d_out)

    return {'d_token_emb': d_token_emb, 'd_pos_emb': d_pos_emb}

# Step 99 - create_qkv_projections
def create_qkv_projections(d_model, d_head, scale=0.02):
    # TODO: return a dict with 'Wq','Wk','Wv', each of shape (d_model, d_head)
    Wq = make_2d_random(d_model, d_head, seed=0) * scale
    Wk = make_2d_random(d_model, d_head, seed=1) * scale
    Wv = make_2d_random(d_model, d_head, seed=2) * scale

    return {'Wq': Wq, 'Wk': Wk, 'Wv': Wv}

# Step 100 - compute_query
import numpy as np

def compute_query(x, w_q):
    """Project x (B, T, d_model) into queries Q (B, T, d_head) using w_q."""
    # TODO: project x into the query space using w_q
    return x @ w_q

# Step 101 - compute_key
def compute_key(x, w_k):
    """Project x through Wk to get keys K of shape (B, T, d_head)."""
    # TODO: project the (B, T, d_model) input through w_k to produce (B, T, d_head) keys.
    return x @ w_k

# Step 102 - compute_value
def compute_value(x, w_v):
    # TODO: project x of shape (B, T, d_model) by w_v of shape (d_model, d_head)
    return x @ w_v

# Step 103 - compute_attention_scores
import numpy as np

def compute_attention_scores(q, k):
    """Return raw attention scores Q @ K^T with shape (B, T, T)."""
    # TODO: compute raw attention scores Q @ K^T per batch element
    return q @ k.swapaxes(-1,-2)

# Step 104 - scale_attention_scores
import numpy as np

def scale_attention_scores(scores, d_head):
    """Rescale (B, T, T) attention scores by a function of d_head."""
    # TODO: rescale the scores so their variance does not grow with d_head.
    return scores / np.sqrt(d_head)

# Step 105 - build_causal_mask
import numpy as np

def build_causal_mask(seq_len):
    """Return a (seq_len, seq_len) boolean lower-triangular mask."""
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))

# Step 106 - apply_causal_mask
import numpy as np

def apply_causal_mask(scaled_scores, causal_mask):
    """Replace future positions in scaled_scores with -inf using causal_mask."""
    return np.where(causal_mask, scaled_scores, -np.inf)

# Step 107 - softmax_attention_weights
import numpy as np

def softmax_attention_weights(masked_scores):
    """Row-wise stable softmax over the last axis of (B, T, T) scores."""
    shifted = masked_scores - masked_scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=-1, keepdims=True)

# Step 108 - attention_weighted_values
import numpy as np

def attention_weighted_values(attn, v):
    """Combine attention weights with values: out = attn @ V.

    attn: (B, T, T) softmaxed attention weights
    v:    (B, T, d_head) value vectors
    returns: (B, T, d_head)
    """
    return attn @ v

# Step 109 - apply_output_projection
import numpy as np

def apply_output_projection(attn_out, w_o):
    """Project attention output (B,T,d_head) through Wo (d_head,d_model)."""
    return attn_out @ w_o

# Step 110 - output_projection_backward
def output_projection_backward(d_proj, cache):
    """Backprop through proj = attn_out @ w_o. Return {'d_attn_out', 'dw_o'}."""
    attn_out = cache['attn_out']
    w_o = cache['w_o']

    d_attn_out = d_proj @ w_o.T
    
    B, T, d_head = attn_out.shape
    attn_out_2d = attn_out.reshape(-1, d_head)        # (B*T, d_head)
    d_proj_2d = d_proj.reshape(-1, d_proj.shape[-1])  # (B*T, d_model)
    dw_o = attn_out_2d.T @ d_proj_2d                  # (d_head, d_model)

    return {'d_attn_out': d_attn_out, 'dw_o':dw_o}

# Step 111 - attention_value_backward
import numpy as np

def attention_value_backward(d_attn_out, cache):
    """Backprop through out = attn @ V.

    d_attn_out: (B, T, d_head) upstream gradient w.r.t. attention output.
    cache: dict with 'attn' of shape (B, T, T) and 'v' of shape (B, T, d_head).
    Returns dict with 'd_attn' (B, T, T) and 'd_v' (B, T, d_head).
    """
    attn = cache['attn']
    v = cache['v']

    # forward: out = attn @ v
    # d_attn = d_out @ v.T
    d_attn = d_attn_out @ v.swapaxes(-1, -2)      # (B,T,T) @ (B,T,d_head).T → (B,T,T)
    # d_v = attn.T @ d_out
    d_v = attn.swapaxes(-1, -2) @ d_attn_out      # (B,T,T).T @ (B,T,d_head) → (B,T,d_head)

    return {'d_attn': d_attn, 'd_v': d_v}

# Step 112 - masked_softmax_backward
import numpy as np

def masked_softmax_backward(d_attn, cache):
    """Backprop through the masked row-wise softmax.

    d_attn: ndarray of shape (B, T, T) -- gradient w.r.t. attention weights.
    cache: dict with 'attn' (B,T,T) and 'causal_mask' (T,T) boolean.
    Returns d_masked_scores of shape (B, T, T).
    """
    attn = cache['attn']
    causal_mask = cache['causal_mask']

    d_masked_scores = attn * (d_attn - (d_attn * attn).sum(axis=-1, keepdims=True))

    d_masked_scores = np.where(causal_mask, d_masked_scores, 0.0)
    return d_masked_scores

# Step 113 - scale_scores_backward
import numpy as np

def scale_scores_backward(d_scaled_scores, d_head):
    """Backprop through the 1/sqrt(d_head) attention score scaling."""
    return d_scaled_scores / np.sqrt(d_head)

# Step 114 - qk_scores_backward
import numpy as np

def qk_scores_backward(d_scores, cache):
    """Backprop through scores = Q @ K^T.

    d_scores: (B, T, T)
    cache: dict with 'q' and 'k', each (B, T, d_head)
    returns: {'d_q': (B, T, d_head), 'd_k': (B, T, d_head)}
    """
    q = cache['q']
    k = cache['k']
    d_q = d_scores @ k
    d_k = d_scores.swapaxes(-1,-2) @ q

    return {'d_q': d_q, 'd_k': d_k}

# Step 115 - qkv_projection_backward
def qkv_projection_backward(d_q, d_k, d_v, cache):
    # backprop through Q=x@Wq, K=x@Wk, V=x@Wv to get dx and dw_q, dw_k, dw_v.
    x = cache['x']
    w_q = cache['w_q']
    w_k = cache['w_k']
    w_v = cache['w_v']

    B, T, d_model = x.shape
    x_2d = x.reshape(-1, d_model)
    x_2d = x.reshape(-1, d_model)
    d_q_2d = d_q.reshape(-1, d_q.shape[-1])
    d_k_2d = d_k.reshape(-1, d_k.shape[-1])
    d_v_2d = d_v.reshape(-1, d_v.shape[-1])

     # dW = X.T @ dY
    dw_q = x_2d.T @ d_q_2d    # (d_model, d_head)
    dw_k = x_2d.T @ d_k_2d
    dw_v = x_2d.T @ d_v_2d

    dx = d_q @ w_q.T + d_k @ w_k.T + d_v @ w_v.T

    return {'dx': dx, 'dw_q': dw_q, 'dw_k': dw_k, 'dw_v': dw_v}

# Step 116 - choose_attention_head_config
def choose_attention_head_config(d_model, n_heads):
    """Return a config dict {'n_heads', 'd_head', 'd_model'} for multi-head attention."""
    # TODO: split d_model into n_heads equal-sized d_head chunks and return the config dict
    if d_model % n_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
    d_head = d_model // n_heads
    return {'n_heads': n_heads, 'd_head': d_head, 'd_model': d_model}

# Step 117 - create_multihead_qkv_projections
def create_multihead_qkv_projections(d_model, scale=0.02):
    """Initialize Wq, Wk, Wv as (d_model, d_model) matrices for multi-head attention."""
    # TODO: build a dict with keys 'Wq', 'Wk', 'Wv', each a scaled (d_model, d_model) random matrix
    w_q = make_2d_random(d_model, d_model, 0)
    w_k = make_2d_random(d_model, d_model, 1)
    w_v = make_2d_random(d_model, d_model, 2)

    scaled_w_q = scale_w_small(w_q, scale)
    scaled_w_k = scale_w_small(w_k, scale)
    scaled_w_v = scale_w_small(w_v, scale)

    return {'Wq': scaled_w_q, 'Wk': scaled_w_k, 'Wv': scaled_w_v}

# Step 118 - create_multihead_output_projection
def create_multihead_output_projection(d_model, scale=0.02):
    """Initialize Wo of shape (d_model, d_model) for multi-head attention output projection."""
    # TODO: build a (d_model, d_model) random matrix and scale it down by `scale`.
    Wo = make_2d_random(d_model, d_model, 0)

    return scale_w_small(Wo, scale)

# Step 119 - reshape_to_heads
import numpy as np

def reshape_to_heads(x, n_heads, d_head):
    """Reshape (B, T, d_model) into (B, T, n_heads, d_head)."""
    # TODO: split the last dimension of x into n_heads chunks of size d_head
    B, T, d_model = x.shape
    return x.reshape(B, T, n_heads, d_head)

# Step 120 - transpose_heads_to_front
import numpy as np

def transpose_heads_to_front(x_heads):
    """Transpose (B, T, n_heads, d_head) to (B, n_heads, T, d_head)."""
    # TODO: move the heads axis in front of the time axis
    return x_heads.transpose(0, 2, 1, 3)

# Step 121 - get_multihead_n_heads
def get_multihead_n_heads(config):
    # TODO: return the number of attention heads stored in the multi-head config dict.
    return config['n_heads']

# Step 122 - get_multihead_sequence_length
import numpy as np

def get_multihead_sequence_length(x):
    """Return T from x of shape (B, T, d_model)."""
    # TODO: return the sequence length T from the (B, T, d_model) tensor.
    return x.shape[1]

# Step 123 - compute_d_head
def compute_d_head(d_model, n_heads):
    # TODO: return the per-head dimension d_head for multi-head attention.
    if d_model % n_heads != 0:
        raise ValueError("n_heads does not evenly divide d_model.")
    return d_model // n_heads

# Step 124 - multihead_masked_softmax_scores
def multihead_masked_softmax_scores(scores, mask):
    """Apply causal mask and row-wise softmax to multi-head attention scores.

    Args:
        scores: ndarray of shape (B, n_heads, T, T)
        mask:   ndarray of shape (T, T), True where positions are kept

    Returns:
        weights: ndarray of shape (B, n_heads, T, T)
    """
    # TODO: mask future positions then row-wise softmax over the last axis
    masked = apply_causal_mask(scores, mask)
    shifted = masked - masked.max(axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=-1, keepdims=True)

# Step 125 - multihead_weighted_sum
import numpy as np

def multihead_weighted_sum(weights, v_heads):
    """Compute per-head attention output as weights @ V across all heads."""
    # TODO: combine attention weights with values across heads
    return weights @ v_heads

# Step 126 - transpose_heads_to_back
def transpose_heads_to_back(x_heads):
    # TODO: move the heads axis back so the result has shape (B, T, n_heads, d_head).
    return x_heads.transpose(0, 2, 1, 3)

# Step 127 - get_multihead_output_sequence_length
def get_multihead_output_sequence_length(x_heads_back):
    """Return T from a (B, T, n_heads, d_head) tensor."""
    # TODO: read the sequence-length dimension from x_heads_back's shape
    return x_heads_back.shape[1]

# Step 128 - merge_heads_to_d_model
import numpy as np

def merge_heads_to_d_model(x_heads_back):
    """Reshape (B, T, n_heads, d_head) into (B, T, d_model)."""
    # TODO: collapse the last two axes into a single d_model axis
    B, T, n_heads, d_head = x_heads_back.shape
    return x_heads_back.reshape(B, T, n_heads * d_head)

# Step 129 - multihead_output_projection_forward
def multihead_output_projection_forward(merged, w_out, b_out):
    """Project the merged multi-head output through the output linear layer.

    Inputs:
      merged: (B, T, d_model)
      w_out:  (d_model, d_model)
      b_out:  (d_model,)
    Returns dict with keys {'out', 'cache'}; cache holds {'merged', 'w_out'}.
    """
    # TODO: project merged through w_out, add b_out, and stash inputs in the cache.
    out = merged @ w_out + b_out
    cache = {'merged': merged, 'w_out': w_out}
    return {'out': out, 'cache': cache}

# Step 130 - multihead_reshape_transpose_backward
def multihead_reshape_transpose_backward(d_merged, shape_info):
    """Invert merge_heads_to_d_model to recover (B, n_heads, T, d_head) gradients."""
    # TODO: undo the merge/transpose/reshape chain from the forward pass
    B = shape_info['B']
    n_heads = shape_info['n_heads']
    T = shape_info['T']
    d_head = shape_info['d_head']

    d_transposed = d_merged.reshape(B, T, n_heads, d_head)
    return d_transposed.transpose(0, 2, 1, 3)

# Step 131 - ffn_linear_one_forward
def ffn_linear_one_forward(x, w1, b1):
    """First FFN linear: lift (B, T, d_model) up to (B, T, d_ff) and add bias."""
    # TODO: apply the first FFN linear that expands d_model to d_ff
    linear_out = linear_forward(x, w1)
    h1 = bias_add_forward(linear_out['y'], b1)
    cache = {
        'x': x,
        'w1': w1,
    }

    return {'h1': h1['y'], 'cache': cache}

# Step 132 - ffn_activation_forward
def ffn_activation_forward(h1):
    """Apply ReLU to FFN hidden pre-activations.

    Args:
        h1: ndarray of shape (B, T, d_ff)

    Returns:
        a1: ndarray of shape (B, T, d_ff)
        cache: dict with key 'h1'
    """
    relu_out = relu_forward(h1)
    a1 = relu_out['y']
    cache = {'h1': h1}
    return a1, cache

# Step 133 - ffn_linear_two_forward
def ffn_linear_two_forward(a1, w2, b2):
    # TODO: project a1 (B, T, d_ff) down to (B, T, d_model) using w2 and b2, return h2 and cache
    linear_out = linear_forward(a1, w2)
    h2 = bias_add_forward(linear_out['y'], b2)
    cache = {
        'a1': a1,
        'w2': w2,
    }

    return {'h2': h2['y'], 'cache': cache}

# Step 134 - ffn_backward
def ffn_backward(d_out, cache):
    """Backprop through linear2 -> ReLU -> linear1 of the FFN.

    cache keys: 'x', 'w1', 'h1', 'a1', 'w2'.
    Returns dict with keys: 'dx', 'dw1', 'db1', 'dw2', 'db2'.
    """
    x = cache['x']
    w1 = cache['w1']
    h1 = cache['h1']
    a1 = cache['a1']
    w2 = cache['w2']

    B, T, d_model = x.shape
    d_ff = a1.shape[-1]

    d_out_2D = d_out.reshape(-1, d_out.shape[-1])
    a1_2d = a1.reshape(-1, d_ff)
    h1_2d = h1.reshape(-1, d_ff)
    x_2d = x.reshape(-1, d_model)

    da1_2d = linear_backward_dx(d_out_2D, {'w': w2})
    dw2 = linear_backward_dw(d_out_2D, {'x': a1_2d})
    db2 = bias_add_backward_db(d_out_2D, {})

    dh1_2d = relu_backward(da1_2d, {'x': h1_2d})

    db1 = bias_add_backward_db(dh1_2d, {})
    dw1 = linear_backward_dw(dh1_2d, {'x': x_2d})
    dx_2d = linear_backward_dx(dh1_2d, {'w': w1})

    dx = dx_2d.reshape(B, T, d_model)

    return {'dx': dx, 'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}

# Step 135 - residual_forward
def residual_forward(x, sublayer_out):
    """Return x + sublayer_out for a residual connection."""
    # TODO: add the sublayer output to its input to form a residual connection.
    return x + sublayer_out

# Step 136 - residual_backward
def residual_backward(d_y):
    """Backprop through y = x + sublayer_out. Returns (d_x, d_sublayer_out)."""
    # TODO: route the upstream gradient to both branches of the residual add.
    d_x = d_y.copy()
    d_sublayer_out = d_y.copy()
    return (d_x, d_sublayer_out)

# Step 137 - pre_layernorm_sublayer_forward
def pre_layernorm_sublayer_forward(x, ln_params, sublayer_fn, sublayer_params):
    # TODO: apply LayerNorm to x, run sublayer_fn on the result, then residual-add back to x.
    ln_out = layernorm_forward_affine(x, ln_params['gamma'], ln_params['beta'], ln_params.get('eps', 1e-5))
    sublayer_out = sublayer_fn(ln_out['y'], sublayer_params)

    y = residual_forward(x, sublayer_out['y'])

    cache = {
        'x': x,
        'ln_cache': ln_out['cache'],
        'sublayer_cache': sublayer_out['cache']
    }

    return {'y': y, 'cache': cache}

# Step 138 - transformer_block_forward
# ── Helper  multihead_attention_forward ──
def multihead_attention_forward(x, attn_params):
    Wq = attn_params['Wq']
    Wk = attn_params['Wk']
    Wv = attn_params['Wv']
    Wo = attn_params.get('Wo', attn_params.get('w_o'))
    b_o = attn_params.get('b_o', attn_params.get('bo', np.zeros(Wo.shape[1])))
    n_heads = attn_params['n_heads']

    B, T, d_model = x.shape
    d_head = compute_d_head(d_model, n_heads)

    # project → (B, T, d_model)
    q = compute_query(x, Wq)
    k = compute_key(x, Wk)
    v = compute_value(x, Wv)

    # split into heads → (B, n_heads, T, d_head)
    q_heads = transpose_heads_to_front(reshape_to_heads(q, n_heads, d_head))
    k_heads = transpose_heads_to_front(reshape_to_heads(k, n_heads, d_head))
    v_heads = transpose_heads_to_front(reshape_to_heads(v, n_heads, d_head))

    # scaled dot-product attention
    scores = scale_attention_scores(compute_attention_scores(q_heads, k_heads), d_head)
    attn_weights = multihead_masked_softmax_scores(scores, build_causal_mask(T))

    # weighted sum → merge → output projection
    context = multihead_weighted_sum(attn_weights, v_heads)
    merged = merge_heads_to_d_model(transpose_heads_to_back(context))
    proj = multihead_output_projection_forward(merged, Wo, b_o)

    cache = {
        'x': x,
        'q_heads': q_heads, 'k_heads': k_heads, 'v_heads': v_heads,
        'attn': attn_weights, 'causal_mask': build_causal_mask(T),
        'context': context, 'merged': merged,
        'w_q': Wq, 'w_k': Wk, 'w_v': Wv, 'w_o': Wo,
        'proj_cache': proj['cache'],
        'n_heads': n_heads, 'd_head': d_head, 'B': B, 'T': T,
    }
    return {'y': proj['out'], 'cache': cache}


# ── Helper  ffn_forward ──
def ffn_forward(x, ffn_params):
    w1, b1 = ffn_params['w1'], ffn_params['b1']
    w2, b2 = ffn_params['w2'], ffn_params['b2']

    h1_out = ffn_linear_one_forward(x, w1, b1)
    a1, _ = ffn_activation_forward(h1_out['h1'])
    h2_out = ffn_linear_two_forward(a1, w2, b2)

    cache = {
        'x': x,
        'w1': w1, 'h1': h1_out['h1'],
        'a1': a1,
        'w2': w2,
    }
    return {'y': h2_out['h2'], 'cache': cache}


# ── Step 138  transformer_block_forward ──
def transformer_block_forward(x, block_params):
    """Run one pre-LN Transformer block forward.

    Args:
        x: ndarray of shape (B, T, d_model).
        block_params: dict with keys 'ln1', 'attn', 'ln2', 'ffn'.

    Returns:
        dict with 'y' (B, T, d_model) and 'cache' with keys
        'attn_branch' and 'ffn_branch'.
    """
    attn_out = pre_layernorm_sublayer_forward(
        x, block_params['ln1'], multihead_attention_forward, block_params['attn']
    )
    ffn_out = pre_layernorm_sublayer_forward(
        attn_out['y'], block_params['ln2'], ffn_forward, block_params['ffn']
    )
    return {'y': ffn_out['y'], 'cache': {'attn_branch': attn_out['cache'], 'ffn_branch': ffn_out['cache']}}

# Step 139 - transformer_block_backward
def transformer_block_backward(d_y, cache, block_params):
    """Backward pass for a pre-LN Transformer block.

    Args:
        d_y: upstream gradient w.r.t. block output, shape (B, T, D).
        cache: dict from transformer_block_forward, with keys 'attn_branch' and 'ffn_branch'.
        block_params: nested dict with keys 'ln1', 'attn', 'ln2', 'ffn'.

    Returns:
        (d_x, grads) where d_x has shape (B, T, D) and grads is a nested dict
        with keys 'ln1', 'ln2', 'attn', 'ffn' mirroring block_params.
    """
    # Tip: recover x from cache['attn_branch']['x'] and call _complete_block_cache(x, block_params)
    # to guarantee every field the backward helpers need is present, no matter what the forward saved.
    # TODO: reverse the FFN branch then the attention branch, summing residual + sublayer gradients
    x = cache['attn_branch']['x']
    full_cache = _complete_block_cache(x, block_params)

    # FFN branch backward
    d_h1_ffn, d_sublayer_ffn = residual_backward(d_y)
    d_z_ffn, ffn_grads = _ffn_sublayer_backward(d_sublayer_ffn, full_cache['ffn_branch']['sublayer_cache'], block_params['attn'])
    d_ln2, d_gamma2, d_beta2 = layernorm_backward_affine(d_z_ffn, full_cache['ffn_branch']['ln_cache'])
    d_h1 = d_h1_ffn + d_ln2

    # Attention branch backward
    d_x_attn, d_sublayer_attn = residual_backward(d_h1)
    d_z_attn, attn_grads = _attn_sublayer_backward(d_sublayer_attn, full_cache['attn_branch']['sublayer_cache'], block_params['attn'])
    d_ln1, d_gamma1, d_beta1 = layernorm_backward_affine(d_z_attn, full_cache['attn_branch']['ln_cache'])
    d_x = d_x_attn + d_ln1

    grads = {
        'ln1': {'gamma': d_gamma1, 'beta': d_beta1},
        'ln2': {'gamma': d_gamma2, 'beta': d_beta2},
        'attn': attn_grads,
        'ffn': ffn_grads
    }
    return (d_x, grads)

# Step 140 - stack_transformer_blocks
import numpy as np

def stack_transformer_blocks(n_layers, d_model, n_heads, d_ff):
    """Build a list of n_layers Transformer block parameter dicts.

    Each block dict has keys 'ln1', 'attn', 'ln2', 'ffn'.
    """
    blocks = []
    seed = 0
    scale = 0.02
    for _ in range(n_layers):
        block = {
            'ln1': {
                'gamma': np.ones(d_model),
                'beta': np.zeros(d_model)
            },
            'attn': {
                'Wq': scale_w_small((make_2d_random(d_model, d_model, seed)), scale),
                'Wk': scale_w_small((make_2d_random(d_model, d_model, seed)), scale),
                'Wv': scale_w_small((make_2d_random(d_model, d_model, seed)), scale),
                'Wo': scale_w_small((make_2d_random(d_model, d_model, seed)), scale),
                'bo': np.zeros(d_model)
            },
            'ln2': {
                'gamma': np.ones(d_model),
                'beta': np.zeros(d_model)
            },
            'ffn': {
                'W1': scale_w_small(make_2d_random(d_model, d_ff, seed), scale),
                'b1': np.zeros(d_ff),
                'W2': scale_w_small(make_2d_random(d_ff, d_model, seed), scale),
                'b2': np.zeros(d_model)
            }
        }
        blocks.append(block)
    
    return blocks

# Step 141 - forward_through_all_blocks
def forward_through_all_blocks(x, blocks):
    """Run x through every Transformer block in order, collecting caches."""
    # TODO: thread x through each block in `blocks`, collecting per-block caches
    caches = []
    y = x
    for block_params in blocks:
        out = transformer_block_forward(y, block_params)
        y = out['y']
        caches.append(out['cache'])
    return y, caches

# Step 142 - backward_through_all_blocks
def backward_through_all_blocks(d_y, caches, blocks):
    """Backprop through a stack of Transformer blocks.

    Inputs:
      d_y     : (B, T, d_model) upstream gradient at the top of the stack
      caches  : list of per-block forward caches
      blocks  : list of per-block parameter dicts

    Returns:
      d_x        : (B, T, d_model) gradient at the input of the stack
      grads_list : list of per-block parameter-gradient dicts, in block order
    """
    # TODO: walk the blocks in reverse, calling transformer_block_backward each step.
    grad_list = []
    d_x = d_y
    for cache, block_params in zip(reversed(caches), reversed(blocks)):
      d_x, grads = transformer_block_backward(d_x, cache, block_params)
      grad_list.append(grads)
    grad_list.reverse()
    return d_x, grad_list

# Step 143 - final_layernorm_forward
def final_layernorm_forward(x, gamma, beta):
    """Apply LayerNorm to a (B, T, d_model) tensor with affine params gamma, beta.

    Returns (y, cache) where cache has keys 'x', 'mean', 'var', 'x_hat', 'gamma'.
    """
    # TODO: normalize each (b, t) position across the d_model channels, then apply gamma/beta.
    eps = 1e-5
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    y = gamma * x_hat + beta

    cache = {
        'x': x,
        'mean': mean,
        'var': var,
        'x_hat': x_hat,
        'gamma': gamma,
    }

    return y, cache

# Step 144 - lm_head_linear_forward
def lm_head_linear_forward(x, w_lm, b_lm):
    """Project hidden states (B,T,d_model) to logits (B,T,vocab_size)."""
    # TODO: project final hidden states to vocab-size logits via the language model head.
    lin_out = linear_forward(x, w_lm)
    bias_out = bias_add_forward(lin_out['y'], b_lm)

    return {
        'logits': bias_out['y'],
        'cache': {
            'x': lin_out['cache']['x'],
            'w_lm': lin_out['cache']['w'],
        }
    }

# Step 145 - full_model_forward
def full_model_forward(x_ids, model_params):
    """Run embeddings, all blocks, final LN, and LM head; return logits and caches."""
    # TODO: chain token+positional embeddings, Transformer blocks, final LayerNorm, and LM head.
    # 1. embeddings
    tok_emb_out, tok_cache = token_embedding_forward(x_ids, model_params['tok_emb'])
    T = x_ids.shape[1]
    pos_emb = slice_positional_embedding(model_params['pos_emb'], T)
    x = add_token_and_positional_embeddings(tok_emb_out, pos_emb)
    emb_cache = {'tok_cache': tok_cache, 'pos_emb': pos_emb, 'seq_len': T}

    # 2. transformer blocks
    x, block_caches = forward_through_all_blocks(x, model_params['blocks'])

    # 3. final layer norm
    ln_f = model_params['ln_f']
    x, ln_f_cache = final_layernorm_forward(x, ln_f['gamma'], ln_f['beta'])

    # 4. language model head
    lm = model_params['lm_head']
    lm_out = lm_head_linear_forward(x, lm['w_lm'], lm['b_lm'])

    caches = {
        'emb': emb_cache,
        'blocks': block_caches,
        'ln_f': ln_f_cache,
        'lm_head': lm_out['cache'],
    }
    return lm_out['logits'], caches

# Step 146 - full_model_backward
def full_model_backward(d_logits, caches, model_params):
    """Propagate d_logits back through LM head, final LN, blocks, and embeddings.

    Args:
        d_logits: (B, T, V) gradient w.r.t. the model output
        caches: nested dict from full_model_forward with keys
                'emb', 'blocks', 'ln_f', 'lm_head'
        model_params: nested dict matching the forward's parameter tree

    Returns:
        grads: nested dict mirroring model_params with keys
               'tok_emb', 'pos_emb', 'blocks', 'ln_f': {'gamma', 'beta'},
               'lm_head': {'w_lm', 'b_lm'}
    """
    B, T, V = d_logits.shape

    # 1. LM head linear backward  (logits = x @ w_lm + b_lm)
    x_lm = caches['lm_head']['x']                          # (B, T, d_model)
    w_lm = caches['lm_head']['w_lm']                       # (d_model, V)
    d_logits_2d = d_logits.reshape(-1, V)                  # (B*T, V)
    x_lm_2d    = x_lm.reshape(-1, x_lm.shape[-1])         # (B*T, d_model)
    d_b_lm     = d_logits_2d.sum(axis=0)                   # (V,)
    d_w_lm     = x_lm_2d.T @ d_logits_2d                  # (d_model, V)
    d_ln_f_out = (d_logits_2d @ w_lm.T).reshape(B, T, -1) # (B, T, d_model)

    # 2. Final LayerNorm backward
    x_hat = caches['ln_f']['x_hat']    # (B, T, D)
    var   = caches['ln_f']['var']       # (B, T, 1)
    gamma = caches['ln_f']['gamma']     # (D,)
    eps   = 1e-5
    std   = np.sqrt(var + eps)
    dgamma   = (d_ln_f_out * x_hat).sum(axis=(0, 1))
    dbeta    = d_ln_f_out.sum(axis=(0, 1))
    dx_hat   = d_ln_f_out * gamma
    d_blocks_out = (1.0 / std) * (
        dx_hat
        - dx_hat.mean(axis=-1, keepdims=True)
        - x_hat * (dx_hat * x_hat).mean(axis=-1, keepdims=True)
    )

    # 3. Transformer blocks backward (reverse order)
    d_emb, block_grads = backward_through_all_blocks(
        d_blocks_out, caches['blocks'], model_params['blocks']
    )

    # 4. Embedding sum backward → token emb + positional emb
    d_tok_emb = token_embedding_backward(d_emb, caches['emb']['tok_cache'])

    seq_len    = caches['emb']['seq_len']
    d_pos_emb  = np.zeros_like(model_params['pos_emb'])
    d_pos_emb[:seq_len] = d_emb.sum(axis=0)   # sum over batch

    return {
        'tok_emb': d_tok_emb,
        'pos_emb': d_pos_emb,
        'blocks':  block_grads,
        'ln_f':    {'gamma': dgamma, 'beta': dbeta},
        'lm_head': {'w_lm': d_w_lm, 'b_lm': d_b_lm},
    }

# Step 147 - initialize_adam_moments
import numpy as np

def initialize_adam_moments(model_params):
    """Allocate zeroed Adam first- and second-moment buffers matching model_params."""
    # TODO: walk the nested parameter dict and build parallel (m, v) zero buffers
    def zero_like_trees(tree):
        if isinstance(tree, np.ndarray):
            return np.zeros_like(tree)
        elif isinstance(tree, dict):
            return {k: zero_like_trees(v) for k, v in tree.items()}
        elif isinstance(tree, list):
            return [zero_like_trees(item) for item in tree]
        else:
            return tree
    
    m = zero_like_trees(model_params)
    v = zero_like_trees(model_params)

    return (m, v)

# Step 148 - initialize_adam_step_counter
def initialize_adam_step_counter():
    """Return the initial Adam step counter t."""
    # TODO: return the starting value of the Adam time-step counter.
    return 0

# Step 149 - adam_increment_step
def adam_increment_step(t):
    """Return t + 1 so Adam bias correction sees a positive step."""
    # TODO: return the next Adam step counter value
    return t + 1

# Step 150 - adam_update_first_moment
import numpy as np

def adam_update_first_moment(m, grad, beta1):
    """Return the updated Adam first-moment estimate."""
    # TODO: blend previous moment m with current grad using decay beta1
    return beta1 * m + (1 - beta1) * grad

# Step 151 - adam_update_second_moment
def adam_update_second_moment(v_prev, grad, beta2):
    """Update Adam's second-moment estimate v using squared gradient EMA."""
    # TODO: blend v_prev with the squared gradient using beta2
    return beta2 * v_prev + (1- beta2) * (grad**2)

# Step 152 - adam_bias_correction
def adam_bias_correction(m, v, beta1, beta2, t):
    """Return bias-corrected (m_hat, v_hat) for Adam at step t."""
    # TODO: divide m and v by (1 - beta**t) factors to remove init bias
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    return (m_hat, v_hat)

# Step 153 - adam_parameter_update
import numpy as np

def adam_parameter_update(param, m_hat, v_hat, lr, eps):
    """Apply the Adam update: param - lr * m_hat / (sqrt(v_hat) + eps)."""
    # TODO: return the updated parameter array of the same shape as param.
    return param - lr * m_hat /  (np.sqrt(v_hat) + eps)

# Step 154 - wire_full_training_loop
def wire_full_training_loop(params, train_ids, val_ids, block_size, batch_size, n_steps, lr, betas, eps):
    """Run the full GPT training loop for n_steps and return (updated_params, history)."""
    # TODO: drive sample-batch -> forward -> loss -> backward -> Adam-update for n_steps...
    beta1, beta2 = betas
    rng = np.random.default_rng(0)
    adam_m, adam_v = initialize_adam_moments(params)
    history = []
    t = initialize_adam_step_counter()

    def update_tree(param, grad, m_node, v_node):
        if isinstance(param, np.ndarray):
            m_new = adam_update_first_moment(m_node, grad, beta1)
            v_new = adam_update_second_moment(v_node, grad, beta2)
            m_hat, v_hat = adam_bias_correction(m_new, v_new, beta1, beta2, t)
            param_new = adam_parameter_update(param, m_hat, v_hat, lr, eps)
            return param_new, m_new, v_new
        
        elif isinstance(param, dict):
            new_param, new_m, new_v = {}, {}, {}
            for k in param:
                new_param[k], new_m[k], new_v[k] = update_tree(param[k], grad[k], m_node[k], v_node[k])
            return new_param, new_m, new_v
        elif isinstance(param, list):
            results = [update_tree(p, g, mi, vi) for p, g, mi, vi in zip(param, grad, m_node, v_node)]
            return [r[0] for r in results], [r[1] for r in results], [r[2] for r in results]
        else:
            return param, m_node, v_node

    for step in range(n_steps):
        t = adam_increment_step(t)

        #1. sample batch
        X, Y = get_batch(train_ids, block_size, batch_size, rng)

        #2. forward
        logits, caches = full_model_forward(X, params)

        #3. loss + d_logits
        B, T, V = logits.shape
        probs = stable_softmax_2d_rowwise(logits.reshape(-1, V))
        loss = cross_entropy_loss(probs, Y.flatten())
        d_probs = softmax_cross_entropy_backward(probs, Y.flatten())
        d_logits = d_probs.reshape(B, T, V)

        #4. backward
        grads = full_model_backward(d_logits, caches, params)

        #5. adam update
        params, adam_m, adam_v = update_tree(params, grads, adam_m, adam_v)

        history.append({'step': step, 'train_loss': float(loss)})

    return params, history

# Step 155 - logging_and_validation_loss
def logging_and_validation_loss(params, val_ids, block_size, batch_size, n_eval_batches):
    """Estimate validation cross-entropy loss by averaging over several batches."""
    rng = np.random.default_rng(42)
    total_loss = 0.0

    for _ in range(n_eval_batches):
        X, Y = get_batch(val_ids, block_size, batch_size, rng)
        logits, _ = full_model_forward(X, params)

        B, T, V = logits.shape
        probs = stable_softmax_2d_rowwise(logits.reshape(-1, V))
        loss = cross_entropy_loss(probs, Y.flatten())
        total_loss += float(loss)
    
    return total_loss / n_eval_batches

# Step 156 - encode_prompt
import numpy as np

def encode_prompt(prompt, stoi):
    """Encode a string prompt to an int ndarray of shape (1, T)."""
    ids = encode_string(prompt, stoi)
    return np.array(ids).reshape(1, -1)

# Step 157 - crop_context_to_block_size
def crop_context_to_block_size(context_ids, block_size):
    return context_ids[: , -block_size:]

# Step 158 - forward_to_get_logits
def forward_to_get_logits(params, context_ids):
    """Run the full model forward and return only the logits tensor."""
    # TODO: drive the full Tiny GPT forward pipeline and return logits of shape (1, T, V).
    logits, _ = full_model_forward(context_ids, params)
    return logits

# Step 159 - take_last_position_logits
def take_last_position_logits(logits):
    """Return logits at the final time step with shape (1, vocab_size)."""
    # TODO: slice out the logits at the final time step from a (1, T, V) tensor.
    return logits[:, -1, :]

# Step 160 - apply_temperature
def apply_temperature(logits, temperature):
    """Scale logits by 1/temperature before softmax sampling."""
    return logits/temperature

# Step 161 - top_k_filter
def top_k_filter(logits, k):
    """Return logits with all but the top-k entries per row set to -inf."""
    V = logits.shape[-1]
    if k >= V:
        return logits
    top_k_values = np.sort(logits, axis=-1)[..., -k]
    mask = logits >= top_k_values[..., np.newaxis]
    return np.where(mask, logits, -np.inf)

# Step 162 - softmax_to_probs
def softmax_to_probs(logits):
    """Convert (1, V) logits into a row-wise probability distribution."""
    return stable_softmax_2d_rowwise(logits)

# Step 163 - sample_one_token
def sample_one_token(probs, rng):
    """Sample one token id from probs of shape (1, vocab_size) using rng."""
    return int(rng.choice(probs.shape[-1], p=probs[0]))

# Step 164 - append_token_to_sequence
import numpy as np

def append_token_to_sequence(context_ids, token_id):
    """Append token_id as a new final column to context_ids of shape (1, T)."""
    new_token = np.array([[token_id]])
    return np.concatenate([context_ids, new_token], axis=1)

# Step 165 - generation_loop_for_n_steps
def generation_loop_for_n_steps(params, prompt_ids, n_new_tokens, block_size, temperature, top_k, rng):
    """Iteratively generate n_new_tokens by repeatedly forwarding the cropped context."""
    context = prompt_ids

    for _ in range(n_new_tokens):
        context_cropped = context[:, -block_size:]

        logits, _ = full_model_forward(context_cropped, model_params)

        last_logits = take_last_position_logits(logits)

        last_logits = apply_temperature(last_logits)

        last_logits = top_k_filter(last_logits, top_k)

        probs = softmax_to_probs(last_logits)

        token_id = sample_one_token(probs, rng)

        context = append_token_to_sequence(context, token_id)
    
    return context

# Step 166 - decode_final_sequence
def decode_final_sequence(generated_ids, itos):
    """Decode a (1, T) id tensor into a string using itos."""
    return decode_ids(generated_ids[0].tolist(), itos)

