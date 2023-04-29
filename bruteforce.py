import numpy as np
import itertools


def brute_force(J, H):
    # Generate all possible vectors S containing only +1 and -1
    S_possible = np.array(list(itertools.product([1, -1], repeat=len(H))))

    # Compute the cost function for each vector S and find the minimum
    min_cost = float('inf')
    min_S = None
    for S in S_possible:
        cost = np.dot(np.transpose(S), np.dot(J, S)) + np.dot(H, S)
        if cost < min_cost:
            min_cost = cost
            min_S = S

    return min_cost, min_S
