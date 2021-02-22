

import numpy as np
print(np.__version__)


from numpy.polynomial.legendre import leggauss
from scipy import interpolate

from timeit import default_timer as timer 
from numba import jit, njit, cuda



#returns the lagrange basis of degree n
#with chebychev points as nodes
def get_lagrange_basis(n):
    bases_nodes_val = np.zeros((n,1),dtype=np.float64)
    nodes = np.polynomial.chebyshev.chebpts2(n)
    basis = []
        
    for i in range(n):
        bases_nodes_val[i]=1
        basis.append( interpolate.lagrange(nodes, bases_nodes_val) )
        bases_nodes_val[i]=0

    return basis


'''
computes the matrices for a "reference FEM element"
assumes the interval is -1,1
gets the cheychev points of type 2
gets the gaussian quadrature points and weights
computes the i-th basis func nd its derivative and evals them
computes the sum o
on the quad points
'''  

def compute_one_dimensional_matrices(n):
    #n chebichev points
    #n quadrature points
    
    
    #nodes = np.polynomial.chebyshev.chebpts2(n)
    quad_p, quad_w = leggauss(n)   
    
    #bases_nodes_val = np.zeros((n,1),dtype=np.float64)
    
    B = np.zeros((n,n),dtype=np.float64)
    D = np.zeros((n,n),dtype=np.float64)
    
    lag_basis = get_lagrange_basis(n)
    
    for i in range(n):
        #bases_nodes_val[i]=1
        
        B[i,:] = lag_basis[i](quad_p)
        
        base_i_prime = np.polyder(lag_basis[i])
        D[i,:] = base_i_prime(quad_p)
    
        #bases_nodes_val[i]=0
                
    
    K = np.zeros((n, n), dtype=np.float64)
    M = np.zeros((n, n), dtype=np.float64)
    A = np.zeros((n, n), dtype=np.float64)
        
    
    K = np.einsum('iq, q, jq -> ij', D, quad_w, D)
    M = np.einsum('iq, q, jq -> ij', B, quad_w, B)
    

    
    A = K + M


    return B, K, M, A








# your code here

'''
def compute_3D_matrices(n, K, M):
    
    k1 = np.einsum('il, jm, kn -> ijklmn', K, M, M).reshape((n**3, n**3))
    k2 = np.einsum('il, jm, kn -> ijklmn', M, K, M).reshape((n**3, n**3))
    k3 = np.einsum('il, jm, kn -> ijklmn', M, M, K).reshape((n**3, n**3))
    m1 = np.einsum('il, jm, kn -> ijklmn', M, M, M).reshape((n**3, n**3))
    
    AAA = k1 + k2 + k3 + m1
    return AAA
'''

def matvec_3D_naive(vinput,n, K, M):
    v = vinput.reshape((n,n,n))
    
    k1 = np.einsum('il, jm, kn, lmn -> ijk', K, M, M, v)
    k2 = np.einsum('il, jm, kn, lmn -> ijk', M, K, M, v)
    k3 = np.einsum('il, jm, kn, lmn -> ijk', M, M, K, v)
    m1 = np.einsum('il, jm, kn, lmn -> ijk', M, M, M, v)
    
    u = k1 + k2 + k3 + m1
    return u.reshape((n**3,))




def matvec_3D_matfree(vinput, n, K, M):
    v = vinput.reshape((n,n,n))
    
    vM = v.dot(M) 
    vK = v.dot(K)
    
    MvM = M.dot(vM)
    KvM = K.dot(vM)
    MvK = M.dot(vK)
    
    vk1 = K.dot(MvM)
    vk2 = M.dot(KvM)
    vk3 = M.dot(MvK)
    vm1 = M.dot(MvM)
    
    u = vk1 + vk2 + vk3 + vm1
    
    return u.reshape((n**3,))


@jit
def matvec_3D_matfree_jit(vinput, n, K, M):
    v = vinput.reshape((n,n,n))
    
    u = np.zeros((n,n,n),dtype=np.float64)
    Mv=np.zeros((n,n,n),dtype=np.float64)
    Kv=np.zeros((n,n,n),dtype=np.float64)
    MMv=np.zeros((n,n,n),dtype=np.float64)
    KMv=np.zeros((n,n,n),dtype=np.float64)
    MKv=np.zeros((n,n,n),dtype=np.float64)
    
    
    for k in range(0, n):
        for l in range(0, n):
            for m in range(0, n):
                for aux in range(0, n): 
                    Mv[k,l,m] += M[k,aux]*v[l,m,aux]
                    Kv[k,l,m] += K[k,aux]*v[l,m,aux]
                    
                
    for j in range(0, n):
        for k in range(0, n):
            for l in range(0, n):
                for aux in range(0, n):
                    MMv[j,k,l] += M[j,aux]*Mv[k,l,aux]
                    KMv[j,k,l] += K[j,aux]*Mv[k,l,aux]
                    MKv[j,k,l] += M[j,aux]*Kv[k,l,aux]
                    
    
    for i in range(0, n):
        for j in range(0, n):
            for k in range(0, n):
                for aux in range(0, n):
                    u[i,j,k] += K[i,aux]*MMv[j,k,aux]
                    u[i,j,k] += M[i,aux]*KMv[j,k,aux]
                    u[i,j,k] += M[i,aux]*MKv[j,k,aux]
                    u[i,j,k] += M[i,aux]*MMv[j,k,aux]
                    

                
    return u.reshape((n**3,))
    

@cuda.jit
def matvec_3D_matfree_cuda(vinput, n, K, M):
    v = vinput.reshape((n,n,n))
    
    u = cuda.shared.array(shape=(n, n, n), dtype=float32)
    Mv= cuda.shared.array(shape=(n, n, n), dtype=float32)
    Kv= cuda.shared.array(shape=(n, n, n), dtype=float32)
    MMv=cuda.shared.array(shape=(n, n, n), dtype=float32)
    KMv=cuda.shared.array(shape=(n, n, n), dtype=float32)
    MKv=cuda.shared.array(shape=(n, n, n), dtype=float32)
    
    
    for k in range(0, n):
        for l in range(0, n):
            for m in range(0, n):
                for aux in range(0, n): 
                    Mv[k,l,m] += M[k,aux]*v[l,m,aux]
                    Kv[k,l,m] += K[k,aux]*v[l,m,aux]
                    
                
    for j in range(0, n):
        for k in range(0, n):
            for l in range(0, n):
                for aux in range(0, n):
                    MMv[j,k,l] += M[j,aux]*Mv[k,l,aux]
                    KMv[j,k,l] += K[j,aux]*Mv[k,l,aux]
                    MKv[j,k,l] += M[j,aux]*Kv[k,l,aux]
                    
    
    for i in range(0, n):
        for j in range(0, n):
            for k in range(0, n):
                for aux in range(0, n):
                    u[i,j,k] += K[i,aux]*MMv[j,k,aux]
                    u[i,j,k] += M[i,aux]*KMv[j,k,aux]
                    u[i,j,k] += M[i,aux]*MKv[j,k,aux]
                    u[i,j,k] += M[i,aux]*MMv[j,k,aux]
                    

                
    return u.reshape((n**3,))    
 
#all_n = np.arange(5,50,5)   
all_n = np.arange(10,110,10)
#all_n = np.arange(50,110,10)

naive_file = 'naive_out_times_log.txt'
matfree_file = 'matfree_out_times_log.txt'
matfree_jit_file = 'matfree_jit_out_times_log.txt'

iters = 10


naive_times_n = np.zeros((iters+1,np.size(all_n)), dtype=np.float64)
matfree_times_n = np.zeros((iters+1,np.size(all_n)), dtype=np.float64)
matfree_jit_times_n = np.zeros((iters+1,np.size(all_n)), dtype=np.float64)


for nn in range(0,np.size(all_n)):
	n = all_n[nn]
	print(n)

	naive_times_n[0,nn] = n
	matfree_times_n[0,nn] = n
	matfree_jit_times_n[0,nn] = n
	

	#get the one-dimensional matrices
	B,K,M,A = compute_one_dimensional_matrices(n)

	vec = np.random.uniform(low=-1, high=1, size=(n**3,))
	
	for i in range(iters):

		start = timer()
		matvec_3D_naive(vec, n, K,M  )
		u_naive = naive_times_n[i+1,nn] = (timer() - start)
		print("naive ",n,i,naive_times_n[i+1,nn])
		
		#print("matfree")
		start = timer()
		matvec_3D_matfree(vec, n, K,M  )
		u_matfree = matfree_times_n[i+1,nn] = (timer() - start)
		print("matfree ",n,i,matfree_times_n[i+1,nn])
		

		start = timer()
		matvec_3D_matfree_jit(vec, n, K,M  )
		u_jit = matfree_jit_times_n[i+1,nn] = (timer() - start)
		print("jit ",n,i,matfree_jit_times_n[i+1,nn])


	np.savetxt(naive_file, naive_times_n[:,0:nn+1], delimiter=',')
	np.savetxt(matfree_file, matfree_times_n[:,0:nn+1], delimiter=',')
	np.savetxt(matfree_jit_file, matfree_jit_times_n[:,0:nn+1], delimiter=',')

