
import math
import numpy as np
#import matplotlib.pyplot as plt

from numpy.polynomial.legendre import leggauss
from scipy import interpolate
#from scipy import integrate
#from scipy.interpolate import KroghInterpolator

#from collections import namedtuple

#import sympy as sym
#import sympy.abc


from timeit import default_timer as timer 
from numba import jit, njit, cuda
import time


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
    


'''
simple cuda kernel of the matmult function
every thread calculates an element of each intermediate matrix and synchronises
'''
@cuda.jit
def cuda_matfree3D_ker(n,vd, ud,M,K,Mv,Kv,MMv,KMv,MKv):
    

    
    
    #absolute position of thread in grid
    gx, gy, gz = cuda.grid(3)
    
    
    
    if gx < n and gy < n and gz < n :
    

        for aux in range(0, n): 
            Mv[gx,gy,gz] += M[gx,aux]*vd[gy,gz,aux]
            Kv[gx,gy,gz] += K[gx,aux]*vd[gy,gz,aux]
        
        cuda.syncthreads()


        for aux in range(0, n):
            MMv[gx,gy,gz] += M[gx,aux]*Mv[gy,gz,aux]
            KMv[gx,gy,gz] += K[gx,aux]*Mv[gy,gz,aux]
            MKv[gx,gy,gz] += M[gx,aux]*Kv[gy,gz,aux]


        cuda.syncthreads()

        for aux in range(0, n):
            ud[gx,gy,gz] += K[gx,aux]*MMv[gy,gz,aux]
            ud[gx,gy,gz] += M[gx,aux]*KMv[gy,gz,aux]
            ud[gx,gy,gz] += M[gx,aux]*MKv[gy,gz,aux]
            ud[gx,gy,gz] += M[gx,aux]*MMv[gy,gz,aux]
        
        cuda.syncthreads()
    

'''
wrapper function for the cuda kernel, initialises device memory
and communicated the 1D matrices and the vector
'''
def matvec_3D_matfree_cuda(vinput, n, K, M): 
    v = vinput.reshape((n,n,n))
    u = np.zeros((n,n,n),dtype=np.float64)
    
    vd = cuda.to_device(v)
    Kd = cuda.to_device(K)
    Md = cuda.to_device(M)
    
    ud= cuda.device_array((n,n,n),dtype=np.float64)
    Mv= cuda.device_array((n,n,n),dtype=np.float64)
    Kv= cuda.device_array((n,n,n),dtype=np.float64)
    MMv=cuda.device_array((n,n,n),dtype=np.float64)
    KMv=cuda.device_array((n,n,n),dtype=np.float64)
    MKv=cuda.device_array((n,n,n),dtype=np.float64)
    
    tpb = 8
    bpg = math.ceil(n/8)
    threadsperblock = (tpb,tpb,tpb)
    blockspergrid = (bpg,bpg,bpg)
    cuda_matfree3D_ker[blockspergrid, threadsperblock](n,vd, ud,Md,Kd,Mv,Kv,MMv,KMv,MKv)
    
   
    
    u = ud.copy_to_host()
    
    return u.reshape((n**3,))  
    
    
#initialise the numba functions and compile them
sizen = 4
vv = np.ones((sizen**3))
B,K,M,A = compute_one_dimensional_matrices(sizen) 
u1=matvec_3D_naive(vv, sizen, K,M  )
u2=matvec_3D_matfree_jit(vv, sizen, K,M  )
u3=matvec_3D_matfree_cuda(vv, sizen, K,M  )




all_n = np.arange(5,20,2)

naive_file = 'naive_out_times_log.txt'
matfree_file = 'matfree_out_times_log.txt'
matfree_jit_file = 'matfree_jit_out_times_log.txt'
matfree_cuda_file = 'matfree_cuda_out_times_log.txt'

iters = 10


#naive_times_n = np.zeros((iters+1,np.size(all_n)), dtype=np.float64)
matfree_times_n = np.zeros((iters+1,np.size(all_n)), dtype=np.float64)
matfree_jit_times_n = np.zeros((iters+1,np.size(all_n)), dtype=np.float64)
matfree_cuda_times_n = np.zeros((iters+1,np.size(all_n)), dtype=np.float64)


for nn in range(0,np.size(all_n)):
	n = all_n[nn]
	print(n)

	n#aive_times_n[0,nn] = n
	matfree_times_n[0,nn] = n
	matfree_jit_times_n[0,nn] = n
	

	#get the one-dimensional matrices
	B,K,M,A = compute_one_dimensional_matrices(n)

	#AAA = compute_3D_matrices(n,K,M)
	
	vec = np.random.uniform(low=-1, high=1, size=(n**3,))

	for i in range(iters):



		#start = timer()
		#matvec_3D_naive(vec, n, K,M  )
		#naive_times_n[i+1,nn] = (timer() - start)
		
		start = timer()
		matvec_3D_matfree(vec, n, K,M  )
		matfree_times_n[i+1,nn] = (timer() - start)
		
		
		start = timer()
		matvec_3D_matfree_jit(vec, n, K,M  )
		matfree_jit_times_n[i+1,nn] = (timer() - start)
		
		
		start = timer()
		matvec_3D_matfree_cuda(vec, n, K,M  )
		matfree_cuda_times_n[i+1,nn] = (timer() - start)


	#np.savetxt(naive_file, naive_times_n[:,0:nn], delimiter=',')
	np.savetxt(matfree_file, matfree_times_n[:,0:nn], delimiter=',')
	np.savetxt(matfree_jit_file, matfree_jit_times_n[:,0:nn], delimiter=',')
	np.savetxt(matfree_cuda_file, matfree_cuda_times_n[:,0:nn], delimiter=',')

    

    #start = timer()
    #matvec_3D_matfree(vec, n, K,M  )
    #print(timer() - start)
#
#
    #start = timer()
    #matvec_3D_matfree_jit(vec, n, K,M  )
    #print(timer() - start)
#
    #start = timer()
    #matvec_3D_matfree_cuda(vec, n, K,M  )
    #print(timer() - start)
