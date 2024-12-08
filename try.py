import numpy as np

def generate_lpn_problem(n, m, error_rate):
    s = np.random.randint(0, 2, n)  # Secret vector
    A = np.random.randint(0, 2, (m, n))  # Binary matrix
    noise = np.random.choice([0, 1], size=m, p=[1 - error_rate, error_rate])  
    b = (A @ s) % 2 
    b = (b + noise) % 2  # Adding noise
    return A, b, s

def majority_vote(buckets):
    majority = []
    for bucket in buckets:
        # Perform majority vote for each bit across all rows in this bucket
        column_sums = np.sum(bucket, axis=0)
        majority.append([1 if col_sum > len(bucket) / 2 else 0 for col_sum in column_sums])
    return majority

def bkw_algo(A, b, k, L, num_basis_vectors):
    m, n = A.shape
    s_estimated = np.zeros(n, dtype=int)  # The estimated secret vector
    
    # Step 1: Partition based on last block of b bits
    for i in range(n-1, 0, -1):  # From the last bit to the second bit
        buckets = {}
        
        # Partition the rows of A and b based on the last i bits of b
        for row_idx in range(m):
            prefix = tuple(b[row_idx][max(0, n-i):])  # Get the last i bits of the row's b value
            if prefix in buckets:
                buckets[prefix].append(A[row_idx])
            else:
                buckets[prefix] = [A[row_idx]]
        
        # For each partition, randomly choose one row to zero out the others
        for prefix, rows in buckets.items():
            if len(rows) > 1:
                chosen_row = np.random.choice(len(rows))
                for j in range(len(rows)):
                    if j != chosen_row:
                        A[rows[j]] = 0  # Zero out the row
                        
        # Use majority vote to recover the bit
        bit_buckets = [bucket for bucket in buckets.values() if len(bucket) > 0]
        majority_bit = majority_vote(bit_buckets)[0][0]
        s_estimated[i] = majority_bit  # Estimate the ith bit of the secret

        # Adjust A and b for the next iteration
        A = A[:, :-i]  # Remove the last i columns
        b = b[:, :-i]  # Remove the corresponding bits of b

    # Step 2: Basis vector recovery (direct zeroing to create standard basis vectors)
    for i in range(num_basis_vectors):
        for j in range(n - 1, 0, -1):
            # Partitioning and zeroing out rows to create standard basis vector
            buckets = {}
            for row_idx in range(m):
                prefix = tuple(b[row_idx][max(0, n-j):])
                if prefix in buckets:
                    buckets[prefix].append(A[row_idx])
                else:
                    buckets[prefix] = [A[row_idx]]
            
            # Zero out the rows except the one corresponding to the bit
            for prefix, rows in buckets.items():
                if len(rows) > 1:
                    chosen_row = np.random.choice(len(rows))
                    for j in range(len(rows)):
                        if j != chosen_row:
                            A[rows[j]] = 0
            bit_buckets = [bucket for bucket in buckets.values() if len(bucket) > 0]
            majority_bit = majority_vote(bit_buckets)[0][0]
            s_estimated[j] = majority_bit
        
    return s_estimated

# Example usage:
n = 8  # size secret vector
m = 65536  # Num of equations
error_rate = 0.1  # Noise rate
k = 2  # bits to partition
L = 4  #a size
num_basis_vectors = 5  # Number of basis vectors to recover

# Generate LPN problem instance
A, b, s = generate_lpn_problem(n, m, error_rate)

# Run BKW algorithm
estimated_secret = bkw_algo(A, b, k, L, num_basis_vectors)

print("Estimated secret vector:")
print(estimated_secret)
