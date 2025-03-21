# TODO: import dependencies and write unit tests below
import numpy as np

from nn.io import read_text_file, read_fasta_file
from nn.nn import NeuralNetwork
from nn.preprocess import one_hot_encode_seqs,  sample_seqs   

# ChatGPT used for assistance with generating example network and tests

nn_arch = [
    {"input_dim": 2, "output_dim": 3, "activation": "relu"},
    {"input_dim": 3, "output_dim": 1, "activation": "sigmoid"}
]

def test_single_forward():
    nn = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=1, epochs=10, loss_function="mse")
    W = np.random.randn(3, 2)
    b = np.random.randn(3, 1)
    A_prev = np.random.randn(2, 1)
    activation = "relu"
    
    A_curr, Z_curr = nn._single_forward(W, b, A_prev, activation)
    
    assert A_curr.shape == (3, 1)
    assert Z_curr.shape == (3, 1)  

def test_forward():
    nn = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=1, epochs=10, loss_function="mse")
    X = np.random.randn(2, 1)
    output, cache = nn.forward(X)
    
    assert output.shape == (1, 1)
    assert isinstance(cache, dict)

def test_single_backprop():
    nn = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=1, epochs=10, loss_function="mse")
    W = np.random.randn(3, 2)
    b = np.random.randn(3, 1)
    Z = np.random.randn(3, 1)
    A_prev = np.random.randn(2, 1)
    dA_curr = np.random.randn(3, 1)
    activation = "relu"
    
    dA_prev, dW, db = nn._single_backprop(W, b, Z, A_prev, dA_curr, activation)
    
    assert dA_prev.shape == (2, 1)
    assert dW.shape == (3, 2)
    assert db.shape == (3, 1)

def test_predict():
    nn = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=1, epochs=10, loss_function="mse")
    X = np.random.randn(2, 1)
    y_hat = nn.predict(X)
    
    assert y_hat.shape == (1, 1)

def test_binary_cross_entropy():
    nn = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=1, epochs=10, loss_function="bce")
    y = np.array([[1]])
    y_hat = np.array([[0.9]])
    loss = nn._binary_cross_entropy(y, y_hat)
    
    assert isinstance(loss, float)
    assert loss > 0

def test_binary_cross_entropy_backprop():
    nn = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=1, epochs=10, loss_function="bce")
    y = np.array([[1]])
    y_hat = np.array([[0.9]])
    dA = nn._binary_cross_entropy_backprop(y, y_hat)
    
    assert dA.shape == y.shape

def test_mean_squared_error():
    nn = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=1, epochs=10, loss_function="mse")
    y = np.array([[1]])
    y_hat = np.array([[0.9]])
    loss = nn._mean_squared_error(y, y_hat)
    
    assert isinstance(loss, float)
    assert loss >= 0

def test_mean_squared_error_backprop():
    nn = NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=1, epochs=10, loss_function="mse")
    y = np.array([[1]])
    y_hat = np.array([[0.9]])
    dA = nn._mean_squared_error_backprop(y, y_hat)
    
    assert dA.shape == y.shape


def test_sample_seqs():
    # Test for balanced
    seqs_bal = ["A", "T", "G", "C", "A", "G", "T", "C"]
    labels_bal = [True, False, True, False, True, False, True, False]
    
    sampled_seqs_bal, sampled_labels_bal = sample_seqs(seqs_bal, labels_bal)

    # The output should be balanced
    assert sampled_labels_bal.count(True) == sampled_labels_bal.count(False)
    assert len(sampled_seqs_bal) == len(sampled_labels_bal)
    assert set(sampled_seqs_bal).issubset(set(seqs_bal))

    # Now test for imbalanced
    seqs_imbal = ["A", "T", "G", "C", "A", "G"]
    labels_imbal = [True, True, True, False, False, False]

    sampled_seqs_imbal, sampled_labels_imbal = sample_seqs(seqs_imbal, labels_imbal)

    # The output should be balanced
    assert sampled_labels_imbal.count(True) == sampled_labels_imbal.count(False)
    assert len(sampled_seqs_imbal) == len(sampled_labels_imbal)

def test_one_hot_encode_seqs():
    # test given proper imput
    seqs = ["ATGC", "CGTA"]
    expected_output = np.array([
        [1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 0, 1,  0, 0, 1, 0],  # ATGC
        [0, 0, 1, 0,  0, 0, 0, 1,  0, 1, 0, 0,  1, 0, 0, 0]   # CGTA
    ])

    output = one_hot_encode_seqs(seqs)

    assert output.shape == expected_output.shape
    np.testing.assert_array_equal(output, expected_output)

    # test empty input
    seqs_empty = []
    output_empty = one_hot_encode_seqs(seqs_empty)

    assert output_empty.shape == (0,)