import numpy as np

# This fucntion takes the solution X (decision VECTOR) and the exchange rate matrix. It raises an error if the solution is invalid
def check_valid_solution(X, exchange_rate_matrix):
    # number of nodes
    V = len(exchange_rate_matrix)
    # number of edges (pairs tradable)
    E = len(exchange_rate_matrix[~np.isnan(exchange_rate_matrix)])

    # First check: the right size
    if len(X) != V**2:
        raise ValueError(f"Decison vector does not have the right shape: lengh of {len(X)} and should be {V**2}")

    # Match the shape of X with the exchange rate matrix
    X = X.reshape(V, V)

    # Second check: first condition
    for i in range(V):
        sum1, sum2 = 0, 0
        for j in range(V):
            if X[i,j] != 0:
                sum1 += X[i,j]
            if X[j,i] != 0:
                sum2 += X[j,i]
    
        if sum1 != sum2:
            raise ValueError(f"The trade sequence is not a circle. Increase coefficient lambda1 and retry")
    
    # Third check: second condition
    for i in range(V):
        sum = 0
        for j in range(V):
            if X[i, j] != 0:
                sum += X[i,j]
        if sum>1:
            raise ValueError(f"Traded the same asset twice. Increase coefficient lambda2 and retry")
    
    print("Solution is valid!")
    
    return

# This function takes the decision VECTOR and log exchange rates matrix and outputs the proffit the arbitrage would have made
def compute_theorical_proffit(X, log_echange_rate_matrix):
    n = len(log_echange_rate_matrix)
    X = X.reshape(n, n)
    
    indexes = np.where(X==1)

    # compute the sum of the logs of the exchange rates
    log_rates_sum = np.sum(log_echange_rate_matrix[X==1])

    return np.exp(log_rates_sum)