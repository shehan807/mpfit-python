import numpy as np
from mpfit import svdcmp_sp, pythag

def test_svd():
    """
    Test the svdcmp_sp function with a simple test matrix.
    """
    # Create a simple test matrix
    # This is a 3x3 matrix with known singular values
    a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=np.float32)
    
    # Create arrays for w and v
    m, n = a.shape
    w = np.zeros(n, dtype=np.float32)
    v = np.zeros((n, n), dtype=np.float32)
    
    # Make a copy of a for the test
    a_test = a.copy()
    
    # Run the SVD decomposition
    print("Running SVD decomposition on test matrix:")
    print(a_test)
    print()
    
    a_result, w_result, v_result = svdcmp_sp(a_test, w, v)
    
    # Print the results
    print("\nResults of SVD decomposition:")
    print("U matrix (stored in a):")
    print(a_result)
    print("\nSingular values (w):")
    print(w_result)
    print("\nV matrix:")
    print(v_result)
    
    # Verify the decomposition
    # U * diag(w) * V^T should be close to the original matrix
    w_diag = np.diag(w_result)
    reconstructed = np.dot(a_result, np.dot(w_diag, v_result.T))
    
    print("\nReconstructed matrix (U * diag(w) * V^T):")
    print(reconstructed)
    
    print("\nDifference from original matrix:")
    print(a - reconstructed)
    
    print("\nMaximum absolute difference:", np.max(np.abs(a - reconstructed)))

if __name__ == "__main__":
    test_svd() 