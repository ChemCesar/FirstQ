#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 08:29:39 2024

@author: cesar
"""

''' 
This files includes codes for various functions that are needed 
for simulating the FSL method. 
'''

import numpy as np

from qiskit import QuantumCircuit, Aer, transpile
import graycode
import cmath
import matplotlib.pyplot as plt

# Qiskit 
from qiskit.circuit.library import QFT

# Custom libraries


def Fourier_state(f,m):
    
    r'''
    This function calculates the (m+1)-qubit
    quantum states |c> encoding the first 2^(m+1) 
    Fourier modes of the function f.
    
    Note that the this state is defined in Eq.(1)
    of our paper.
        
    Args:
        f (list): Target state of a function 
            whose Fourier state we need to prepare.
        m (int): Determines the number of Fourier
            modes that we need to determine.
            
    Returns:
        c_state (list): Quantum state of m+1 qubits
            with 2^{m+1} dominant Fourier coefficients.
            
    Raises ValueError:
        If m is not less than total number of qubits.
    '''
    
    if len(f) < 2**(m+1):
        raise ValueError("m should be less than total number of qubits.")
    
    # Fourier Coefficients
    c = np.fft.ifft(f)

    # Fourier state:
    c_state = [*c[:2**m],0,*c[len(f)+1-2**m:]]
    c_state = c_state/np.linalg.norm(c_state)
    
    return c_state

def Fourier_state_2d(f,m):
    
    r'''
    This function calculates the 2*(m+1)-qubit
    quantum states |c> encoding the first 2^(2m+2) 
    Fourier modes of a function f of two variables.
    
    Note that the this state is defined in Eq.(C3)
    of our paper.
        
    Args:
        f (list): Target state of a 2d function 
            whose Fourier state we need to prepare.
        m (int): Determines the number of Fourier
            modes that we need to determine.
            
    Returns:
        c_state (list): Quantum state of 2*(m+1) qubits
            with 2^{2m+2} dominant Fourier coefficients.
            
    Raises ValueError:
        If m is not less than total number of qubits.
    '''
    
    N = len(f)
    if N < 2**(m+1):
        raise ValueError("m should be less than total number of qubits per dimension.")
    
    # Fourier Coefficients
    c = np.fft.ifft2(f)

    # Fourier state:
    c_state = []
    
    for k in range(2**m):
        c_state.extend([*c[k][:2**m],0,*c[k][N+1-2**m:]])
    c_state.extend(np.zeros(2**(m+1)))
    for k in range(N+1-2**m,N):
        c_state.extend([*c[k][:2**m],0,*c[k][N+1-2**m:]])
            
    c_state = c_state/np.linalg.norm(c_state)
    
    return c_state

def output_reordering(qiskit_output):
    
    '''
    The output from the Qiskit uses a different qubit
    ordering convention from the convention that we used.
    For example, Qiskit labels the state |01000> as |2>
    and |00010> as |16>. On the other hand, our convention
    demands that |2> should be a label of |00010> and |16> 
    be a label of |01000>. 
    
    This function takes an output from the Qiskit and
    returns the output which is consistent with our
    convention.
    
    Args:
        qiskit_output (list): Output from the simulation
            performed by Qiskit. 
    '''
    
    # We first determine how many qubits there are.
    # Note: dim(qiskit_output) = 2^n
    n = int(np.round(np.log(len(qiskit_output))/np.log(2)))
    
    new_output = []
    
    for k in range(len(qiskit_output)):
        
        bin_k = bin(k)[2:]
        bin_k = (n-len(bin_k))*'0' + bin_k
        bin_k_rev = bin_k[::-1]
        k_rev = int(bin_k_rev,2)
        
        new_output.append(qiskit_output[k_rev])
    
    return new_output   


# UCR = Uniformly controlled Rotations
def cascade_UCRs(target_state, with_phases = True):
    
    r'''
    This function returns a QuantumCircuit implementing the cascade
    of uniformly controlled rotations. This circuit can be used to 
    map an all-zero state to a target state. 
    
    Args:
        target_state (list): the state vector that we need to prepare.
        with_phases (boolean): True by default. If chosen to be False, 
            we will only get a circuit that prepares the state
            \sum_{k} |psi_k| \ket{k} instead of the full state 
            \sum_{k} \psi_{k} \ket{k}. It is especially desirable when 
            all the amplitudes are real and positive. 
    
    Returns:
        circuit (QuantumCircuit): that maps all-zero state to the
            target_state.
    
    Return type:
        Qiskit QuantumCircuit
            (https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.html)
    
    Raises ValueError if: 
        the length of the target state is not 2^integer.
    '''
    
    # Determine the no. of qubits:
    no_qubits = int(np.round(np.log(len(target_state))/np.log(2)))
    if len(target_state) != 2**no_qubits:
        raise ValueError("The length of the target state should be 2^integer.")
    
    
    # Calculate the angles of rotations from the target state.
    # If the target_state is real and positive, then the angles
    # of RZ rotations are not needed.
    thetas_y = find_thetas_y(target_state)
    if with_phases:
        thetas_z, global_phase = find_thetas_z(target_state)
    
    # Initialize the quantum circuit.
    circuit = QuantumCircuit(no_qubits)
    qubits = [k for k in range(no_qubits)]
    
    # First apply the global phase:
    if with_phases:
        circuit.rz(-global_phase,qubits[0])
    # Then apply uniformly controlled RY:
    for q in range(no_qubits):
        qs = qubits[:(q+1)]
        UCRs(circuit, 'y', thetas_y[no_qubits-1-q], qs)
    # Then apply uniformly controlled RZ:
    if with_phases:
        for q in range(no_qubits):
            qs = qubits[:(q+1)]
            UCRs(circuit, 'z', thetas_z[no_qubits-1-q], qs)
    
    return circuit

def UCRs(circuit, axes, thetas, qubits):
    
    ''' 
    This function implements the decompositon of a single uniformly
    controlled rotation in terms of rotation gates and CNOT gates.
    
    Args:
        circuit (QuantumCircuit in Qiskit): the circuit in which we 
            need to add a layer of uniformly controlled rotation.
        axes (str): axes for rotations. Either 'y' or 'z' are allowed.
        thetas (list): list of angles of rotations.
        qubits (list): list of qubits on which the uniformly controlled
            rotations are to be applied.
    
    Raises ValueError if: 
        axes other that 'y' or 'z' is chosen.
    '''
    
    if axes == 'y':
        rot = circuit.ry
    elif axes == 'z':
        rot = circuit.rz
    else:
        raise ValueError("Only y or z axes are allowed.")
    
    qubits.reverse()
    
    # The control qubit for the last CNOT (if there is any) is the last qubit. 
    if len(thetas)>1:
        circuit.cx(qubits[-1],qubits[0])
    rot(thetas[len(thetas)-1],qubits[0])
    
    # The control qubit for all other CNOTs are determined in terms of graycodes.
    # This is determined using the function cnot_position.
    rs = [(len(thetas)-2-r) for r in range(len(thetas)-1)]
    for r in rs:
        circuit.cx(qubits[cnot_position(r+1)],qubits[0])
        rot(thetas[r],qubits[0])
        

def find_thetas_y(target):
    
    '''
    This function finds the angles of RY rotations that appear 
    in the decomposition of the uniformly controlled rotations.
    
    Note that the angles thetas are related to angles alphas
    according to a linear map: 
        thetas = M * alphas (where M is a matrix)
    Therefore, we first find alphas and then map them to thetas.
    
    Args: 
    target (list): the state vector we need to prepare.
    
    Output: (nested list) angles of rotations.
    '''
    
    alphas_y = find_alphas_y(target)
    thetas_y = thetas_from_alphas(alphas_y)
    
    return thetas_y
    
def find_thetas_z(target):
    
    '''
    This function finds the angles of RZ rotations that appear 
    in the decomposition of the uniformly controlled rotations.
    
    Note that the angles thetas are related to angles alphas
    according to a linear map: 
        thetas = M * alphas (where M is a matrix)
    Therefore, we first find alphas and then map them to thetas.
    
    Args: 
    target (list): the state vector we need to prepare.
    
    Output: (nested list) angles of rotations.
    '''
    
    alphas_z, phase = find_alphas_z(target)
    thetas_z = thetas_from_alphas(alphas_z)
    
    return thetas_z, phase

def find_alphas_y(target):
    
    r'''
    This function find the angles of RY rotations that appear 
    in the uniformly controlled rotations.
    
    These angles can be calculated from the absolute value of
    the target state vector (psi_0, psi_1, ... , psi_{2^n - 1}):
    
    \alpha_y[j][k] = 2 \arcsin(ratio) ,
    ratio = numerator/denominator ,
    numerator = \sum_{\ell=0}^{2^{j}-1} |\psi_{(2k+1)2^{j}+\ell}|^{2} ,
    denominaor = \sum_{\ell=0}^{2^{j+1}-1} |\psi_{k 2^{j+1}+\ell}|^{2} .
    
    Args: 
    target (list): the state vector we need to prepare.
    
    Output: (nested list) angles of rotations.
    '''

    n = int(np.log(len(target))/np.log(2))

    alphas_y = []

    for j in range(n):
        alpha_j = []

        for k in range(2**(n-j-1)):

            num = 0
            for l in range(2**j):
                num = num + np.abs(target[(2*k+1)*(2**(j))+l])**2
            num = np.sqrt(num)

            den = 0
            for l in range(2**(j+1)):
                den = den + np.abs(target[k*(2**(j+1))+l])**2
            den = np.sqrt(den)

            if (den<num):
                raise ValueError("Argument of arcsin has to be less than 1.")
            elif den==num:
                ratio = 1
            else:
                ratio = num/den

            alpha_j.append(2*np.arcsin(ratio))

        alphas_y.append(alpha_j)

    return alphas_y
    
def find_alphas_z(target):
    
    '''
    This function find the angles of RZ rotations that appear 
    in the uniformly controlled rotations.
    
    These angles can be calculated from the phases of the
    target state vector (\omega_0, \omega_1, ... , \omega_{2^n - 1}):
    
    \alpha_z[j][k] = 2^{-j} \sum_{\ell=0}^{2^{j}} (\omega_{(2k+1)2^{j}+\ell} - \omega_{k 2^{j+1}+\ell}) ,
    
    and
    
    global_phase = 2^{1-n} \sum_{j=0}^{2^{n}-1} \omega_{j} .
    
    Args: 
    target (list): the state vector we need to prepare.
    
    Output: (nested list) angles of rotations.
    '''

    n = int(np.log(len(target))/np.log(2))
    phases = [cmath.phase(t) for t in target]
    
    alphas_z = []

    for j in range(n):
        alpha_j = []

        for k in range(2**(n-j-1)):

            sum = 0
            for l in range(2**j):
                sum = sum + (phases[(2*k+1)*(2**j)+l]-phases[(2*k)*(2**j)+l])
            sum = sum/(2**j)

            alpha_j.append(sum)

        alphas_z.append(alpha_j)

    global_phase = 2*np.mean(phases)

    return alphas_z, global_phase


def thetas_from_alphas(alphas):

    '''
    This function implements the linear map M between angles alpha's
    and angles thetas:
                         thetas = M * alphas
    
    Args: 
        alphas (nested list): angles of rotation in the undecomposed 
            uniformly controlled rotations.
    
    Output:
        thetas (nested list): angles of rotation in the decomposed 
            uniformly controlled rotations.
    '''

    thetas = []

    for alpha in alphas:

        theta = []
        
        for i in range(len(alpha)):
            theta_i = 0

            for j in range(len(alpha)):
                theta_i = theta_i + M(i,j)*alpha[j]
            theta_i = theta_i/len(alpha)

            theta.append(theta_i)

        thetas.append(theta)

    return thetas

def M(i,j):

    r'''
    This function calculates the matrix elements of M = (-1)^{g_i \cdot b_j},
    where g_i abd b_i are the gray code and the binary code of the integer i
    respectively. 
    
    Args: 
        i (int): column of the matrix M
        j (int): row of the matrix M
    
    Output:
        M_ij (float): (i,j)^th matrix element of M.
    '''

    bj = bin(j)[2:]
    bj_rev = bj[::-1]
    gi = bin(graycode.tc_to_gray_code(i))[2:]
    gi_rev = gi[::-1]

    mij = 0
    for x,y in zip(bj_rev,gi_rev):
        mij = mij + int(x)*int(y)

    return (-1)**mij

def cnot_position(r):
    
    '''
    This function calculates the position of the control qubit for CNOTs 
    in a decomposed uniformly controlled rotations. 
    
    The position of the r^th CNOT (except for the last CNOT) is determined
    based on where the gray code of the interger r differs from that of
    integer r-1.  
    
    Args: 
        r (int): number of CNOT
    
    Output (integer): position of the r^th CNOT.
    '''

    g1 = bin(graycode.tc_to_gray_code(r))[2:]
    g2 = bin(graycode.tc_to_gray_code(r-1))[2:]

    if len(g2)<len(g1):
        g2 = '0' + g2

    g1_rev = g1[::-1]
    g2_rev = g2[::-1]

    for p in range(len(g1)):
        if g1_rev[p] != g2_rev[p]:
            return p+1


def run_FSL_1d(f,m,n_operators=1):
    # # no. of qubits 
    n = m
    m = n_operators
    # Discretizing the domain [0,1] into 2^n uniformly spaced points
    x = [k/(2**n) for k in range(2**n)]

    #Qiskit quantum state vector simulator
    simulator = Aer.get_backend('statevector_simulator')
    # Target function
    target_f = [f(xx) for xx in x]
    target_f = target_f/np.linalg.norm(target_f)
    # No. of Fourier modes
    # is 2^(m+1)

    # |c> with first 2^(m+1)
    # Fourier modes
    state_c = Fourier_state(target_f,m)
    U_c = cascade_UCRs(state_c)
    # Initialize a quantum circuit
    circ = QuantumCircuit(n)

    # Apply U_c on the last (m+1) qubits
    Uc_qubits = range(n-m-1,n)
    circ.compose(U_c,qubits=Uc_qubits,inplace=True)

    # Apply a cascade of CNOTs
    for i in range(n-m-1):
        circ.cx(n-m-1,i)

    # Finally, apply the inverse QFT.
    inv_qft = QFT(num_qubits=n, inverse=True)
    inv_qft = transpile(inv_qft, simulator)
    # Note that Qiskit uses the opposite 
    # qubit ordering. 
    circ.compose(inv_qft,qubits=range(n-1,-1,-1),inplace=True)
    
    # Transpile the circuit for the simulator
    transpiled_circ = transpile(circ, simulator)
    
    # Print the transpiled circuit size and depth
    print("Transpiled Circuit size:", transpiled_circ.size())
    print("Transpiled Circuit depth:", transpiled_circ.depth())
    # Run the circuit
    # and get the statevector
    job = simulator.run(circ)
    output_state = job.result().get_statevector(circ)
    # Reordering qubits
    simulated_f = output_reordering(output_state)
    simulated_f = [simulated_f[i].real for i in range(len(simulated_f))]
    infidelity = 1 - np.abs(np.dot(np.conjugate(simulated_f),target_f))**2
    print('Infidelity=', infidelity)
    
    return simulated_f, target_f, transpiled_circ.size(), transpiled_circ.depth()


def from_list_to_array(L):
    n = int(len(L) ** 0.5)
    return np.array(L).reshape((n, n))

def from_array_to_list(A):
    return A.flatten().tolist()

def run_FSL_2d(f,m,n_operators=1):
    n = m//2
    
    x = [k/(2**m) for k in range(2**m)]
    target_f = [f(xx) for xx in x]
    target_f = target_f/np.linalg.norm(target_f)
    
    target_f = from_list_to_array(target_f)
    m = n_operators
    
    state_c = Fourier_state_2d(target_f,m)
    U_c = cascade_UCRs(state_c)
    
    # Initialize a quantum circuit
    circ = QuantumCircuit(2*n)
    
    # Apply U_c on (n-m-1, ... , n-1) 
    # and (2n-m-1, ... , 2n -1 ) qubits
    Uc_qubits = [*[_ for _ in range(n-m-1,n)],*[_ for _ in range(2*n - m -1,2*n)]]
    circ.compose(U_c,qubits=Uc_qubits,inplace=True)
    # Apply a cascade of CNOTs
    for i in range(n-m-1):
        circ.cx(n-m-1,i)
        circ.cx(2*n-m-1,n+i)
    # Applying the inverse QFTs
    # on the first n qubits and
    # the last n qubits.
    inv_qft = QFT(num_qubits=n, inverse=True)
    # Note that Qiskit uses the opposite 
    # qubit ordering. 
    circ.compose(inv_qft,qubits=range(n-1,-1,-1),inplace=True)
    circ.compose(inv_qft,qubits=range(2*n-1,n-1,-1),inplace=True)

    # simulator
    simulator = Aer.get_backend('statevector_simulator')
    
    # Transpile the circuit for the simulator
    transpiled_circ = transpile(circ, simulator)
    
    # Print the transpiled circuit size and depth
    print("Transpiled Circuit size:", transpiled_circ.size())
    print("Transpiled Circuit depth:", transpiled_circ.depth())
    # Run the circuit
    # and get the statevector
    circ = transpile(circ, simulator)
    job = simulator.run(circ)
    output_state = job.result().get_statevector(circ)
    
    # Changing qubits ordering
    # convention and reshaping
    # the result.
    simulated_f = output_reordering(output_state)
    simulated_f = [simulated_f[i].real for i in range(len(simulated_f))]
    # simulated_f = np.array(simulated_f).reshape((2**n,2**n))
    target_f = target_f.flatten().tolist()
    infidelity = 1 - np.abs(np.dot(np.conjugate(simulated_f),target_f))**2
    print(infidelity)
    
    return simulated_f, target_f, transpiled_circ.size(), transpiled_circ.depth()































