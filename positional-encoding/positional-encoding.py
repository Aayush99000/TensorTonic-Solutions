import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    PE = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len)[:, np.newaxis]
    i   = np.arange(0, d_model, 2)[np.newaxis, :]

    angles = pos / np.power(base, i / d_model)  # (seq_len, ceil(d_model/2))

    PE[:, 0::2] = np.sin(angles)
    PE[:, 1::2] = np.cos(angles[:, :d_model // 2])  # trim if d_model is odd

    return PE