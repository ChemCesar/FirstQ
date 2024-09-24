# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 14:08:34 2024

@author: JZylberman

Quantum circuit for diagonal operators approximated by Fourier Series using Generalized Quantum Signal Processing
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
from qiskit import QuantumRegister, QuantumCircuit
import qiskit.quantum_info as qi
from qiskit import Aer, transpile, execute

"""test functions"""

def gaussian2(x):
    #sigma=0.1
    sigma=0.15
    mu=0.5
    return(np.exp(-(x-mu)**2/(2*sigma**2))/(4*sigma))

def sinc(x):
    return(np.sinc(np.pi*6*(x-0.5)))

def s(x):
    return(np.sin(2*np.pi*(x)))

"""Fourier coefficients"""

def fourier_coeff(f,j,N): #Fourier coeff of exp(if)
    c=0
    for i in range(N):
        c+=f(i/N)*np.exp(-1j*2*np.pi*i*j/N)
    return(c/N)

def list_fourier_coeff_discrete(f,N,n):
    L=[]
    for j in range(-N,N):
        L.append(fourier_coeff(f,j,n))
    return(L)

def fourier_coeff_continuous(f,j):
    def f_bis(x):
        return(f(x)*np.exp(-1j*j*2*np.pi*x))
    I=scipy.integrate.quad(f_bis, 0, 1, epsabs=1.49e-08, epsrel=1.49e-08, limit=100,complex_func=True)
    return(I[0])

def list_fourier_coeff_continuous(f,N):
    L=[]
    for j in range(-N,N):
        L.append(fourier_coeff_continuous(f,j))
    return(L)

"""GQSP routines"""


def complementary(poly, N):
    """Algorithm 1 to compute the complementary polynomial
    Parameters:
    poly : length (d+1) vector of monomial coefficients of P(z)
    N : int : size of the FFT, N >= (d+1)
    Returns:
    length (d+1) vector of monomial coefficients of Q(z)
    """
    # Pad P to FFT dimension N
    paddedPoly = np.zeros(N, dtype=np.complex128)
    paddedPoly[:poly.shape[0]] = poly

    # Evaluate P(omega) at roots of unity omega
    pEval = np.fft.ifft(paddedPoly, norm="ortho")

    # Compute log(1-|P(omega)|^2) at roots of unity omega
    theLog = np.log(1 - np.abs(pEval)**2)

    # Apply Fourier multiplier in Fourier space
    modes = np.fft.fft(theLog, norm="ortho")
    modes[0] *= 0.5  # Note: modes are ordered differently in the text
    modes[N//2+1:] = 0

    theLog = np.fft.ifft(modes, norm="ortho")

    # Compute coefficients of Q
    coefs = np.fft.fft(np.exp(theLog), norm="ortho")

    # Truncate to length of Q polynomial
    q = coefs[:poly.shape[0]]

    return q


def rotation_angle(p_coeff,q_coeff,d):
    S=np.array([p_coeff,q_coeff],dtype=complex)
    a=p_coeff[d-1]
    b=q_coeff[d-1]
    th=np.arctan(abs(b/a))
    phi=np.angle(a/b)
    if d==1:
        lamb=np.angle(b)
        return([th],[phi],lamb)
    L=np.zeros(np.shape(S),dtype=complex)
    for i in range(d):
        L[0][i]=np.exp(-1j*phi)*np.cos(th)*S[0][i]+np.sin(th)*S[1][i]
        L[1][i]=np.exp(-1j*phi)*np.sin(th)*S[0][i]-np.cos(th)*S[1][i]
    p_coeff_new=L[0][1:d]
    q_coeff_new=L[1][0:d-1]
    th_vec,phi_vec,lamb=rotation_angle(p_coeff_new,q_coeff_new,d-1)
    th_vec.insert(len(th_vec),th)
    phi_vec.insert(len(phi_vec),phi)
    return(th_vec,phi_vec,lamb)

"""quantum circuits for GQSP"""

def R(theta,phi,lamb):
    q = QuantumRegister(1,'q')
    qc= QuantumCircuit(q)
    
    if lamb!=0:
        qc.x(0)
        qc.p(lamb,0)
        qc.x(0)
    qc.u(2*theta,0,np.pi,0)
    qc.x(0)
    qc.p(phi,0)
    qc.x(0)
    
    R = qc.to_gate(label="R")
    #print(qc.draw())
    return(R)

def hadamard():
    q = QuantumRegister(1,'q')
    qc= QuantumCircuit(q)
    qc.h(q[0])
    hgate=qc.to_gate(label="hgate")
    return(hgate)
    
def Uw(j,n):
    q = QuantumRegister(n,'q')
    qc= QuantumCircuit(q)
    for i in range(n):
        qc.p(2*np.pi*2**i*j/2**n,i)
    Uwj = qc.to_gate(label="Uwj")
    #print(qc.draw())
    return(Uwj)

def one_step_gqsp(n,theta,phi,lamb):
    q = QuantumRegister(n,'q')
    a = QuantumRegister(1,'a')
    qc= QuantumCircuit(a,q)
    
    rot=R(theta,phi,lamb)
    U=Uw(1,n)
    
    qubit=[i for i in a]
    qc.append(rot,qubit)
    
    U_controlled= U.control(1,ctrl_state=0)
    qubit2=[i for i in a]+[i for i in q]
    qc.append(U_controlled,qubit2)
    #print(qc.draw())
    one_step=qc.to_gate(label="one_step")
    return(one_step)

def gqsp(n,angles):
    theta_vec=angles[0]
    phi_vec=angles[1]
    lamb=angles[2]
    d=len(theta_vec)
    q = QuantumRegister(n,'q')
    a = QuantumRegister(1,'a')
    qc= QuantumCircuit(a,q)
    for j in range(d):
        if j==0:
            one_step=one_step_gqsp(n,theta_vec[j],phi_vec[j],lamb)
        else:
            one_step=one_step_gqsp(n,theta_vec[j],phi_vec[j],0)
        qubit=[i for i in a]+[i for i in q]
        qc.append(one_step,qubit)
    qsvt=qc.to_gate(label="qsvt")
    return(qsvt)

""" quantum circuit for diagonal operator """

def qc_diagonal_1D(f,n,M):
    N=2**n
    # p_coeff=list_fourier_coeff_continuous(f,M)
    p_coeff=list_fourier_coeff_discrete(f,M,N)

    l=len(p_coeff)
    
    def p(x):
        s=0
        for i in range(l):
            s+=p_coeff[i]*x**(i)
        return(s)
    
    m=max(p(1),max([abs(p(np.exp(1j*u))) for u in np.linspace(0,2*np.pi,N)]))
    p_coeff=np.array(p_coeff/(m**2))

    Nbis=1024
    q_coeff=complementary(p_coeff,Nbis)

    angles=rotation_angle(np.array(p_coeff),np.array(q_coeff),len(p_coeff))
    q = QuantumRegister(n,'q')
    a = QuantumRegister(1,'a')
    qc= QuantumCircuit(a,q)

    uw=Uw(-M-1, n)
    u_controlled= uw.control(1,ctrl_state=0)
    
    qubit=[i for i in a]+[i for i in q]
    qc.append(u_controlled,qubit)
    
    qc_gqsp=gqsp(n,angles)
    qc.append(qc_gqsp,qubit)

    diag_fourier=qc.to_gate(label="diag_fourier")
    #print(qc.decompose().decompose().draw())
    return(diag_fourier)

"""verification diagonal operator"""
    
def verification_diag_1D(f,n,M):
    N=2**n
    q = QuantumRegister(n,'q')
    a = QuantumRegister(1,'a')
    qc= QuantumCircuit(a,q)
    
    qubit=[i for i in a]+[i for i in q]
    hgate=hadamard()
    h_controlled= hgate.control(1,ctrl_state=0)
    for i in range(n):
        qubiti=[j for j in a]+[q[i]]
        qc.append(h_controlled,qubiti)
        
    X=np.array([i for i in range(N)])/N
    m=max([f(x) for x in X])
    def f_normalized(x):
        return(f(x)/m)
    
    diag=qc_diagonal_1D(f,n,M)
    qc.append(diag,qubit)
    stv = qi.Statevector.from_instruction(qc)
    L=stv.data
    l=len(L)
    
    
    """ We take values of the vector for which the ancilla qubit is in state 1 and we need to renormalize."""
    
    wavevector=[L[2*i]for i in range(int(l/2))]
    phase_bis=np.angle(wavevector)/np.linalg.norm(np.angle(wavevector))
    Y=[f_normalized(x) for x in X]
    Y=Y/np.linalg.norm(Y)
    plt.plot(X,phase_bis,c='b',ls='-',marker='x',label="Implemented state")
    plt.plot(X,Y,c='r',ls='--',marker='.',label='Target state')
    
    plt.title("diagonal unitary on n="+str(n)+" qubits")
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()

def run_GQSP(f,n,n_operators=1):
    N=2**n
    q = QuantumRegister(n,'q')
    a = QuantumRegister(1,'a')
    qc= QuantumCircuit(a,q)
    
    qubit=[i for i in a]+[i for i in q]
    hgate=hadamard()
    h_controlled= hgate.control(1,ctrl_state=0)
    for i in range(n):
        qubiti=[j for j in a]+[q[i]]
        qc.append(h_controlled,qubiti)
        
    X=np.array([i for i in range(N)])/N
    m=max([f(x) for x in X])
    def f_normalized(x):
        return(f(x)/m)
    
    diag=qc_diagonal_1D(f,n,n_operators)
    qc.append(diag,qubit)
    stv = qi.Statevector.from_instruction(qc)
    L=stv.data
    l=len(L)
    
    simulator = Aer.get_backend('qasm_simulator')
    qc_trans = transpile(qc, simulator)
    circ_trans = transpile(qc, basis_gates=['u','cx','cp','cz'])
    print("Diag Unitary size_brut:", qc.size(),"size_transpiled:",circ_trans.size(),"size_transpiled2:",qc_trans.size())
    print("Diag Unitary depth_brut:", qc.depth(),"depth_transpiled:",circ_trans.depth(), "depth_transpiled:",qc_trans.depth())

    """ We take values of the vector for which the ancilla qubit is in state 1 and we need to renormalize."""
    
    wavevector=[L[2*i]for i in range(int(l/2))]
    # phase_bis=np.angle(wavevector)/np.linalg.norm(np.angle(wavevector))
    wavevector /= np.linalg.norm(wavevector)
    Y=[f_normalized(x) for x in X]
    Y=Y/np.linalg.norm(Y)
    plt.plot(X,wavevector,c='b',ls='-',marker='x',label="Implemented state")
    plt.plot(X,Y,c='r',ls='--',marker='.',label='Target state')
    
    plt.title("diagonal unitary on n="+str(n)+" qubits")
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()
    
    
    infidelity = 1 - abs(np.array(wavevector).T@Y)**2
    print('Infidelity=', infidelity)

    return np.real(wavevector), np.real(Y)
# """tests"""
# verification_diag_1D(gaussian2,10,4)
# verification_diag_1D(sinc,7,15)