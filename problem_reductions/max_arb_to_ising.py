import numpy as np
from scipy.linalg import circulant

def max_arb_to_ising(n, change_rates_martix, lambd1, lambda2):
    log_change_rates_matrix = np.log(change_rates_martix)

    # number of vertices
    V = len(log_change_rates_matrix)
    n=V

    # C tilde
    first_row = np.zeros(n-1)
    first_row[1] = 1
    C = circulant(first_row)
    C_tilde = np.block([[C] * (n-1)])

    # number of edges
    E = np.count_nonzero(np.isnan(log_change_rates_matrix))

    ### partie de J liée a la fonciton de cout sans contrainte
    J = np.outer(log_change_rates_matrix, log_change_rates_matrix)
    # virer les termes diagonaux (sur que ça change rien?)
    np.fill_diagonal(J, 0)

    ### Partie de J liée a la contrainte 1
    J -= np.identity(n)

    for i in range(1, n+1):
        P_i = P_i(n, i)

        # partie 1
        temp = np.zeros((n-1)**2)
        for k in range(0, n):
            temp += np.dot(C_tilde ** k, P_i)
        
        J += 2*np.dot(P_i, temp)

        # partie 2
        for k in range(0, n):
            J += np.dot(P_i, C**k)
    


def P_i(n, i):
    P_i = np.zeros((n-1)**2, (n-1)**2)
    P_i[i-1, i-1] = 1

    return P_i

def Q_i():
    return 