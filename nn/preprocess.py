# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
import pandas as pd

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    # convert to df for easier data handling, separate to pos/neg
    df = pd.DataFrame({'seqs': seqs, 'labels' : labels})
    pos = df[df['labels'] == True]
    neg = df[df['labels'] == False]
    # upsample minority class
    if len(neg) < len(pos):
        neg = neg.sample(len(pos), replace=True)
    else:
        pos = pos.sample(len(neg), replace=True)
    # concat results and reset index
    sampled = pd.concat([pos, neg]).sample(frac=1).reset_index(drop=True)

    return sampled['seqs'].tolist(), sampled['labels'].tolist()


def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    one_hot = {
        'A': [1, 0, 0, 0],
        'T': [0, 1, 0, 0],
        'C': [0, 0, 1, 0],
        'G': [0, 0, 0, 1]
    }

    encodings = []
    for seq in seq_arr:
        seq_map = []
        for nuc in seq:
            seq_map.extend(one_hot[nuc])
        encodings.append(seq_map)

    return np.array(encodings)