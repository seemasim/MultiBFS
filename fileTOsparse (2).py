import numpy as np
import pandas as pd
import scipy.sparse as ss

def read_data_file_as_coo_matrix(filename='soc-LiveJournal1.txt'):
    "Read data file and return sparse matrix in coordinate format."
    data = pd.read_csv(filename, sep='\t', header=None, dtype=np.uint32)
    rows = data[0]  # Not a copy, just a reference.
    cols = data[1]
    ones = np.ones(len(rows), np.uint32)
    matrix = ss.coo_matrix((ones, (rows, cols)))
    matrix = matrix.todok()
    for i in range(1632803):
        matrix[i,i] = 1
    matrix = matrix.tocoo()
    return matrix

def save_csr_matrix(filename, matrix):
    """Save compressed sparse row (csr) matrix to file.

    Based on http://stackoverflow.com/a/8980156/232571

    """
    assert filename.endswith('.npz')
    attributes = {
        'data': matrix.data,
        'indices': matrix.indices,
        'indptr': matrix.indptr,
        'shape': matrix.shape,
    }
    np.savez(filename, **attributes)
    
def load_csr_matrix(filename):
    """Load compressed sparse row (csr) matrix from file.

    Based on http://stackoverflow.com/a/8980156/232571

    """
    assert filename.endswith('.npz')
    loader = np.load(filename)
    args = (loader['data'], loader['indices'], loader['indptr'])
    matrix = ss.csr_matrix(args, shape=loader['shape'])
    return matrix

def test():
    "Test data file parsing and matrix serialization."
    coo_matrix = read_data_file_as_coo_matrix()
    csr_matrix = coo_matrix.tocsr()
    save_csr_matrix('edges.npz', csr_matrix)
    loaded_csr_matrix = load_csr_matrix('edges.npz')
    # Comparison based on http://stackoverflow.com/a/30685839/232571
    #assert (csr_matrix != loaded_csr_matrix).nnz == 0

if __name__ == '__main__':
    test()
