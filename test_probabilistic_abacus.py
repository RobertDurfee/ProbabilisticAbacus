import unittest
import numpy as np
import probabilistic_abacus

class TestProbabilisticAbacus(unittest.TestCase):

    def test1(self):
        transition_matrix = [
            [0, 2, 1, 0, 0],
            [0, 0, 0, 3, 4],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        expected = get_expected(transition_matrix)
        actual = probabilistic_abacus.get_absorption_probabilities(transition_matrix)
        self.assertTrue(np.allclose(np.array(expected), np.array(actual)))

    def test2(self):
        transition_matrix = [
            [0, 1, 0, 0, 0, 1],
            [4, 0, 0, 3, 2, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
        expected = get_expected(transition_matrix)
        actual = probabilistic_abacus.get_absorption_probabilities(transition_matrix)
        self.assertTrue(np.allclose(np.array(expected), np.array(actual)))

    def test3(self):
        transition_matrix = [
            [8, 1, 6, 10, 6],
            [8, 5, 9,  9, 9], 
            [6, 8, 5,  5, 2],
            [0, 0, 0,  0, 0],
            [8, 7, 0,  9, 4],
        ]
        expected = get_expected(transition_matrix)
        actual = probabilistic_abacus.get_absorption_probabilities(transition_matrix)
        self.assertTrue(np.allclose(np.array(expected), np.array(actual)))

    def test4(self):
        transition_matrix = [
            [4, 3, 6, 7, 10],
            [0, 0, 0, 0,  0],
            [7, 8, 6, 2,  2],
            [3, 8, 7, 6,  4],
            [0, 0, 0, 0,  0],
        ]
        expected = get_expected(transition_matrix)
        actual = probabilistic_abacus.get_absorption_probabilities(transition_matrix)
        self.assertTrue(np.allclose(np.array(expected), np.array(actual)))

    def test5(self):
        transition_matrix = [
            [0,  5, 5, 3,  5],
            [4, 10, 9, 8, 10],
            [0,  0, 0, 0,  0],
            [0,  0, 0, 0,  0],
            [4,  9, 0, 0,  9],
        ]
        expected = get_expected(transition_matrix)
        actual = probabilistic_abacus.get_absorption_probabilities(transition_matrix)
        self.assertTrue(np.allclose(np.array(expected), np.array(actual)))

    def test6(self):
        transition_matrix = [
            [0, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
        expected = get_expected(transition_matrix)
        actual = probabilistic_abacus.get_absorption_probabilities(transition_matrix)
        self.assertTrue(np.allclose(np.array(expected), np.array(actual)))

    def test7(self):
        transition_matrix = [
            [0, 1, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 1],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 3, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
        expected = get_expected(transition_matrix)
        actual = probabilistic_abacus.get_absorption_probabilities(transition_matrix)
        self.assertTrue(np.allclose(np.array(expected), np.array(actual)))

def get_markov_matrix(transition_matrix):
    transition_matrix = np.array(transition_matrix)
    transition_matrix_sum = transition_matrix.sum(axis=1, keepdims=True)
    markov_matrix = np.divide(transition_matrix, transition_matrix_sum, out=np.zeros_like(transition_matrix, dtype=np.float32), where=(transition_matrix_sum != 0), dtype=np.float32)
    terminal_states = np.argwhere(transition_matrix_sum.flatten() == 0)
    markov_matrix[terminal_states] = np.eye(markov_matrix.shape[0], dtype=np.float32)[terminal_states]
    return markov_matrix

def get_expected(transition_matrix):
    if len(transition_matrix) == 0:
        return []
    markov_matrix = get_markov_matrix(transition_matrix)
    terminal_states = np.argwhere(np.all(markov_matrix == np.eye(markov_matrix.shape[0]), axis=1)).flatten()
    transient_states = np.argwhere(~np.all(markov_matrix == np.eye(markov_matrix.shape[0]), axis=1)).flatten()
    q = markov_matrix[transient_states][:, transient_states]
    n = np.linalg.inv(np.eye(q.shape[0]) - q)
    r = markov_matrix[transient_states][:, terminal_states]
    b = np.dot(n, r)
    if 0 in transient_states:
        row = np.asscalar(np.argwhere(transient_states == 0))
        cols = np.argsort(terminal_states)
        return b[row][cols].tolist()
    else:
        return np.eye(len(terminal_states))[0].tolist()

if __name__ == '__main__':
    unittest.main()

