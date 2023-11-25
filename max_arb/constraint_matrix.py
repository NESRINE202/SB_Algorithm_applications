#matrice pour la premiere contrainte
import numpy as np

################################# premiere contrainte

# remarque erreur sur rapport premier contriate premiere ligne mdr

# sum_i(sum(sum(Xij1*xij2)))

def matrice_Ci(n):
    C=np.zeros((n*n-n,n*n-n))
    for i in range(n):
        C[i*(n-1):i*(n-1)+n-1,i*(n-1):i*(n-1)+n-1]=1
    return C # X.T*C*X retourne une partie de  cont

# sum_i(sumj1(sumj2(xj1i*xj2i)))

def matrice_P(n): #inverser collone et ligne
    P=np.zeros((n*n-n,n*n-n))
    l=[(n-1)*j for j in range(1,n)]
    for i in range(n):
        for j in range(len(l)):
            P[i*(n-1)+j,l[j]]=1
        for j in range(len(l)):
            if l[j]-(j*n-1)==n:
                l[j]=0
            else:
                l[j]=(l[j]+1)
    return P


def matrice_Cj(n):
    P=matrice_P(n)
    return np.dot(P.T,np.dot(matrice_Ci(n),P))


# sum_i(sumj1(sumj2(xj1i*xij2)))

def matrice_Cj1j2(n):
    return np.dot(matrice_Ci(n),matrice_P(n))

################## deuxieme contrainte

### pareil premiere partie de la 2eme contrainte # sum_i(sum(sum(Xij1*xij2)))

#### deuxieme partie de la 2eme contrainte

def one(n):
    return np.ones(n*n-n)

##### retour

def transaction_fract(matr):
    l = []
    for i in range(np.shape(matr)[0]):
        row = np.log(matr[i, :])
        l.extend(row[:i])
        l.extend(row[i+1:])
    return np.array(l)

def usage(lambda1,lambda2,n,matr):
        
        Contrainte1M=matrice_Ci(n)+matrice_Cj(n)-2*matrice_Cj1j2(n)
        Contrainte2M=matrice_Ci(n)
        M=-lambda1*Contrainte1M-lambda2*Contrainte2M
        H=transaction_fract(matr)-lambda2*one(n)
        print("M utilisé=",M/4)
        
        print("H utilisé=",H/2+np.dot(M,one(n)))
        
        return M,H

if __name__=="__main__":
        
    epsilon=0.01
    matr=np.array([[1,1.1,0.8,1-epsilon],[1/1.1-epsilon,1,0.9,1/0.9-epsilon],[1/0.8-epsilon,1/0.9-epsilon,1,1.1],[1,0.9,1/1.1-epsilon,1]])
    print(usage(10,20,4,matr))
                   
