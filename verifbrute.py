def verif_function (M,H):
    
    import numpy as np
    
    n=M.shape[0]
    
    def decimal_to_binary(k, n):
    # Return a list of length n with 1's and -1's representing the binary digits of k
        binary_str = format(k, 'b').zfill(n)
        return [-1 if bit == '0' else 1 for bit in binary_str]

    def calcule_energie(Matrice,t,H):
        return np.dot(t,np.dot(Matrice,t))+ np.dot(H,t)
    


    configurations = np.array([[-1] * n])
    for i in range(2**n-1):
        binary = np.array(list(bin(i+1)[2:].zfill(n)), dtype=int)
        configurations = np.vstack((configurations, 2*binary-1))

    energies = np.dot(configurations, np.dot(M, configurations.T)).diagonal() + np.dot(configurations, H)

# parcour tt les cas pour avoir le min

    best_t1 = [-1] * n
    min_value=calcule_energie(M,best_t1,H)


    for i in range(2 ** n):
        t1 = decimal_to_binary(i, n)
        m = calcule_energie(M,t1,H)
        if m < min_value:
            min_value = m
            best_t1 = t1

    print(best_t1)
    print(min_value)


    