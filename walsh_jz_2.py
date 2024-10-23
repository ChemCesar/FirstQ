#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 13:35:06 2024

@author: cesar
"""



# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 09:13:37 2023

@author: Julien Zylberman

1D DIAGONAL UNITARY IMPLEMEMTATION THROUGH WALSH OPERATORS
"""


import numpy as np
from bitstring import BitArray

import qiskit.quantum_info as qi
from qiskit import QuantumRegister, QuantumCircuit
from qiskit import Aer, transpile, execute

from Walsh import WalshConstructor

import os

def walsh_operator(j,theta,n,threshold=0): #n number of qubits, a walsh coefficient, X is the register
    """
    Parameters
    ----------
    j : int
        order of the Walsh function.
    theta : float
        angle of the rotation, usually corresponding to the j-Walsh coefficient times a constant.
    n : int
        number of qubits encoding the position axis.

    Returns
    -------
    (qiskit.circuit.gate.Gate,qiskit.circuit.gate.Gate)
    1D j-th Walsh operators

    """
    q = QuantumRegister(n,'q')
    qc= QuantumCircuit(q)
    
    if abs(theta)<=threshold:
        return(qc.to_gate(label="walsh1D"))
    
    lj=bin(j)
    nj=len(lj)-3
    
    for i in range(nj):
        if lj[-i-1]=='1':
            qc.cx(i,nj)
    if j==0:
        qc.x(0)
        qc.p(theta,0)
        qc.x(0)
        qc.p(theta,0)
    else:
        qc.rz(-2*theta,nj)
    
    for i in range(nj):
        if lj[-i-1]=='1':
            qc.cx(i,nj)
    walsh1D = qc.to_gate(label="walsh1D")
    return(walsh1D)
            

def norm(f,N):
    a=0
    K=np.array(range(N))/N
    for i in range(N):
        a+=f(K[i])*np.conjugate(f(K[i]))
    return(np.real(np.sqrt(a)))    
           

def walsh_decreasing_order_qc(L,m,n,threshold=0,n_operators=0): 
    """ 
    Parameters
    ----------
    L : array
        Array of 2**mx*2**my Walsh coeffients/angles
    m : int
        number of qubits on the position register on which is encoding the diagonal unitary.
    nx : int
        number of qubits encoding the x-axis.
    Returns
    -------
    (qiskit.circuit.gate.Gate,qiskit.circuit.gate.Gate)
    1D diagonal unitary associated to L

    """  
    q = QuantumRegister(n,'q')
    qc= QuantumCircuit(q)
    I=np.argsort(abs(np.array(L)))
    # print(I[-n_operators:])
    for j in range(2**m):
        j1=I[-j-1]
        if abs(L[j1])<threshold:
            break
        if j>=n_operators:
            break
        walsh1D=walsh_operator(j1,L[j1],n,threshold)
        qubits = [i for i in q]
        qc.append(walsh1D,qubits)  
    diagonal_unitary_1D = qc.to_gate(label="diagonal_unitary_1D")
    return(diagonal_unitary_1D)

def diagonal_unitary_1D(f,m,n,threshold=0,n_operators=0,method='graycode',swap_option=False,input_walsh_coeff=None):
    """
    Parameters
    ----------
    f : function
        target real function defined on [0,1].
    m : int
        number of qubits on which is implemented the diagonal unitary.
    n : int
        number of qubits.
    swap_option : Boolean, optional
        If Yes, perform the swapping of the quantum register. The default is False.

    Returns
    -------
    (qiskit.circuit.gate.Gate,qiskit.circuit.gate.Gate)
    1D diagonal unitary associated to f
    """
    n_cpus = 1
    script_dir = os.path.dirname(__file__)
    walsh = WalshConstructor(m, n_operators, n_cpus, script_dir, f)
    walsh.parallel_sparse_walsh_series()
    L = walsh.parallel_list_walsh_coeff(walsh.dir_path)

    q = QuantumRegister(n,'q')
    qc = QuantumCircuit(q)
    # L=List_walsh_coeff(f,2**m)
    M=[np.sign(L[i]) for i in range(len(L))]

    if input_walsh_coeff is not None: #if the walsh coefficient list is given as input
        L = [input_walsh_coeff[i] * M[i] for i in range(len(M))]
    
    if swap_option:
        for i in range(int((n)/2)):
            qc.swap(i,n-i-1)
            
    if method=='decreasing_order':
        fullwalsh=walsh_decreasing_order_qc(L,m,n,threshold,n_operators)
        
    qubits = [i for i in q]
    qc.append(fullwalsh,qubits)  
    
    if swap_option:
        for i in range(int((n)/2)):
            qc.swap(i,n-i-1)
            
            
    # simulator = Aer.get_backend('qasm_simulator')
    # qc_trans = transpile(qc, simulator)
    # circ_trans = transpile(qc, basis_gates=['u','cx','cp','cz'])
    # if input_walsh_coeff is None:
    #     print("Diag Unitary size_brut:", qc.size(),"size_transpiled:",circ_trans.size(),"size_transpiled2:",qc_trans.size())
    #     print("Diag Unitary depth_brut:", qc.depth(),"depth_transpiled:",circ_trans.depth(), "depth_transpiled:",qc_trans.depth())
    
    diagonal_unitary = qc.to_gate(label="diagonal_unitary")
    return(diagonal_unitary, L) 

def run_WSL(f,m,n,threshold=0,n_operators=1,method='decreasing_order', swap_option=True, input_walsh_coeff=None):
    
    #____________________________initial circuit depth estimation________________
    q = QuantumRegister(n+1, 'q')
    circuit = QuantumCircuit(q)
    # classical_bits = ClassicalRegister(1)
    # circuit.add_register(classical_bits)
    qubits = [i for i in q]
    
    for i in range(n+1):  # Hadamard tower to initialize
        circuit.h(i)
        
    # Add the custom diagonal unitary
    diag_u, L_walsh_coeff = diagonal_unitary_1D(f,m,n,threshold,n_operators,method,swap_option, input_walsh_coeff=None)

    ctrl_diag_u = diag_u.control(1)
    circuit.append(ctrl_diag_u, qubits)    
    
    # diag_u_dagger = diag_u.inverse()
    # ctrl_diag_u = diag_u_dagger.control(1, ctrl_state='0')
    # circuit.append(ctrl_diag_u, qubits) 

    circuit.h(0)
    circuit.s(0)
    #measurement in Walsh basis
    for i in range(1, n+1):
        circuit.h(i)
            
    # print(circuit.draw())
    qc = circuit
    simulator = Aer.get_backend('qasm_simulator')
    qc_trans = transpile(qc, simulator)
    circ_trans = transpile(qc, basis_gates=['u','cx','cp','cz'])
    print("size_brut:", qc.size(),"size_transpiled:",circ_trans.size(),"size_transpiled2:",qc_trans.size())
    print("depth_brut:", qc.depth(),"depth_transpiled:",circ_trans.depth(), "depth_transpiled:",qc_trans.depth())
    # size, depth = circ_trans.size(), circ_trans.depth()
    size, depth = qc_trans.size(), qc_trans.depth()

    #now real algo________________
    
    q = QuantumRegister(n,'q')
    
    circuit = QuantumCircuit(q)
    qubits = [i for i in q]
    
    for i in range(n): #Hadamard tower to initialize
        circuit.h(i)
        
    diag_u, list_walsh_coeff = diagonal_unitary_1D(f,m,n,threshold,n_operators,method,swap_option, input_walsh_coeff)
    circuit.append(diag_u,qubits)    

    stv = qi.Statevector.from_instruction(circuit)    
    
    wavevector=stv.data
    l=len(wavevector)
    
    X=np.array(range(2**n))/2**n
    
    norm1=0 
    for i in range(l):
        norm1+=np.conjugate(wavevector[i])*wavevector[i]
    norm1=np.sqrt(norm1)
    phase=np.angle(wavevector)
    wavevector=wavevector/norm1
    
    reference = [f(X[i]) for i in range(2**n)]

    reference /= np.linalg.norm(reference)
    trial = phase/np.linalg.norm(phase)
    
    infidelity = 1 - abs(trial.T@reference)
    print('Infidelity=', infidelity)

    return trial, reference, size, depth

def run_WSL_shot(f, m,n, threshold=0, n_operators=20, shots=100000, method='decreasing_order', swap_option=True):
    q = QuantumRegister(n+1, 'q')
    circuit = QuantumCircuit(q)
    # classical_bits = ClassicalRegister(1)
    # circuit.add_register(classical_bits)
    qubits = [i for i in q]
    
    for i in range(n+1):  # Hadamard tower to initialize
        circuit.h(i)
        
    # Add the custom diagonal unitary
    diag_u, L_walsh_coeff = diagonal_unitary_1D(f,m,n,threshold,n_operators,method,swap_option, input_walsh_coeff=None)

    ctrl_diag_u = diag_u.control(1)
    circuit.append(ctrl_diag_u, qubits)    
    
    # diag_u_dagger = diag_u.inverse()
    # ctrl_diag_u = diag_u_dagger.control(1, ctrl_state='0')
    # circuit.append(ctrl_diag_u, qubits) 

    circuit.h(0)
    circuit.s(0)
    #measurement in Walsh basis
    for i in range(1, n+1):
        circuit.h(i)
            
    # print(circuit.draw())
    qc = circuit
    simulator = Aer.get_backend('qasm_simulator')
    qc_trans = transpile(qc, simulator)
    circ_trans = transpile(qc, basis_gates=['u','cx','cp','cz'])
    print("size_brut:", qc.size(),"size_transpiled:",circ_trans.size(),"size_transpiled2:",qc_trans.size())
    print("depth_brut:", qc.depth(),"depth_transpiled:",circ_trans.depth(), "depth_transpiled:",qc_trans.depth())
    # Run using shots (measurement)    
    # Execute the circuit on a QASM simulator with the specified number of shots
    
    circuit.measure_all()
    backend = Aer.get_backend('qasm_simulator')
    job = execute(circuit, backend, shots=shots)
    result = job.result()
    
    counts = result.get_counts(circuit)
    # Count the number of states where the first qubit is 1
    count_first_qubit_1 = sum(count for state, count in counts.items() if state[-1] == '1')
    
    filtered_counts = {}
    for state, count in counts.items():
        if state[-1] == '1':
            suppressed_state = state[:-1]  # Remove the ancilla qubit
            if suppressed_state in filtered_counts:
                filtered_counts[suppressed_state] += count
            else:
                filtered_counts[suppressed_state] = count

    print(f"Number of success: {count_first_qubit_1}")
    assert(count_first_qubit_1>1)
    
    ref_operators = np.argsort(abs(np.array(L_walsh_coeff)))[-n_operators:]
    # Convert counts to a wavevector-like format
    wavevector = np.zeros(2**n, dtype=float)
    for state, count in filtered_counts.items():
        state = state[len(state)//2:][::-1] + state[:len(state)//2][::-1]
        index = int(state, 2)
        wavevector[index] = np.sqrt(count / shots)
    # return print(wavevector)
    
    ref_operators = np.argsort(abs(np.array(L_walsh_coeff)))[-n_operators:]
    selected_operators = np.argsort(abs(np.array(wavevector)))[-n_operators:]
    # print('reference_ops=', ref_operators,'\n','selected_ops=', selected_operators,'\n')
    if set(ref_operators) != set(selected_operators): print("Unexact set of operators, reduce e0 if necessary")  
    
    return run_WSL(f, m, n, 0, n_operators, method, swap_option=True, input_walsh_coeff=wavevector)
