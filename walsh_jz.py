

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


def orbital(x, Ax=0.5, ax=0, alpha=1):
    return (x-Ax)**ax*np.exp(-alpha*(x-Ax)**2)

"""Walsh functions"""

def Walsh(j,x): 
    """
    Parameters
    ----------
    j : int
        Order of the Walsh function.
    x : float
        real number in [0,1].

    Returns
    -------
    The value of j-th Walsh function at position x.
    """
    jbin=bin(j)
    lj=len(jbin)-2
    
    X=dyatic(x,lj)

    p=0
    for i in range(lj):
        p+=int(int(jbin[-1-i])*X[i])
    return((-1)**p)

def dyatic(x,n):
    """
    Parameters
    ----------
    x : float
        real number in [0,1].
    n : int
        index for the truncation of the dyatic expansion of x.

    Returns
    -------
    return the list of coefficient of the dyatic expansion of x up to order n

    """
    L=np.zeros((n))
    a=x
    for i in range(n):
        if a-1/2**(i+1)>=0:
            L[i]=1
            a=a-1/2**(i+1)
    return(L)

def Walsh_coeff(j,f,N,e0):
    """
    Parameters
    ----------
    j : int
        order of the Walsh coefficient.
    f : function
        Function of one variable.
    N : int
        integer representing the number of points on which is computed the Walsh coefficient.
    e0 : float
        Amplifies the signal to improve success probability, must not be too large
    Returns
    -------
    j-th Walsh coeffient of the N-th Walsh series of f.

    """
    K=np.array(range(N))/N
    a=0
    for i in range(N):
        a+=e0*f(K[i])*Walsh(j,K[i])/N
    return(a)


def List_walsh_coeff(f, N, e0):
    return [0 if (abs(a := Walsh_coeff(j, f, N, e0))) < 1e-10 else a for j in range(N)]

def generate_gray_list(my_val):
    """
    Parameters
    ----------
    my_val : int

    Returns
    -------
    List of the first 2**my_val binary numbers in an order such that only one digit differs from one number to the next one.

    """
    if (my_val <= 0):
       return
    my_list = list()
    my_list.append("0")
    my_list.append("1")
    i = 2
    j = 0
    while(True):
       if i >= 1 << my_val:
          break
       for j in range(i - 1, -1, -1):
          my_list.append(my_list[j])
       for j in range(i):
          my_list[j] = "0" + my_list[j]
       for j in range(i, 2 * i):
          my_list[j] = "1" + my_list[j]
       i = i << 1
       
    return(my_list)

""" quantum circuits and operators"""

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
           
def walsh_graycode_qc(G,L,m,n,threshold=0): 
    """ 
    Parameters
    ----------
    G : List
        Gray_list of the 2**n first binary numbers.
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
    
    for j in range(2**m):
        j1=BitArray(bin=G[j]).uint
        walsh1D=walsh_operator(j1,L[j1],n,threshold)
        qubits = [i for i in q]
        qc.append(walsh1D,qubits)  
    
    diagonal_unitary_1D = qc.to_gate(label="diagonal_unitary_1D")
    return(diagonal_unitary_1D)


         
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

def diagonal_unitary_1D(f,m,n,e0,threshold=0,n_operators=0,method='graycode',swap_option=False,input_walsh_coeff=None):
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
    q = QuantumRegister(n,'q')
    qc = QuantumCircuit(q)
    
    L=List_walsh_coeff(f,2**m,e0)
    M=[np.sign(L[i]) for i in range(len(L))]

    if input_walsh_coeff is not None: #if the walsh coefficient list is given as input
        L = [input_walsh_coeff[i] * M[i] for i in range(len(M))]

    if swap_option:
        for i in range(int((n)/2)):
            qc.swap(i,n-i-1)
    
    if method=='graycode':
        G=generate_gray_list(m)
        fullwalsh=walsh_graycode_qc(G,L,m,n,threshold)
    elif method=='decreasing_order':
        fullwalsh=walsh_decreasing_order_qc(L,m,n,threshold,n_operators)
        
    qubits = [i for i in q]
    qc.append(fullwalsh,qubits)  
    
    if swap_option:
        for i in range(int((n)/2)):
            qc.swap(i,n-i-1)
            
            
    simulator = Aer.get_backend('qasm_simulator')
    qc_trans = transpile(qc, simulator)
    circ_trans = transpile(qc, basis_gates=['u','cx','cp','cz'])
    if input_walsh_coeff is None:
        print("Diag Unitary size_brut:", qc.size(),"size_transpiled:",circ_trans.size(),"size_transpiled2:",qc_trans.size())
        print("Diag Unitary depth_brut:", qc.depth(),"depth_transpiled:",circ_trans.depth(), "depth_transpiled:",qc_trans.depth())
    
    diagonal_unitary = qc.to_gate(label="diagonal_unitary")
    return(diagonal_unitary, L) 

def run_WSL(f,m,e0,threshold=0,n_operators=1,method='decreasing_order', swap_option=True, input_walsh_coeff=None):
    q = QuantumRegister(m,'q')
    
    circuit = QuantumCircuit(q)
    qubits = [i for i in q]
    
    for i in range(m): #Hadamard tower to initialize
        circuit.h(i)
        
    diag_u, list_walsh_coeff = diagonal_unitary_1D(f,m,m,e0,threshold,n_operators,method,swap_option, input_walsh_coeff)
    circuit.append(diag_u,qubits)    

    stv = qi.Statevector.from_instruction(circuit)    
    
    wavevector=stv.data
    l=len(wavevector)
    
    X=np.array(range(2**m))/2**m
    
    norm1=0 
    for i in range(l):
        norm1+=np.conjugate(wavevector[i])*wavevector[i]
    norm1=np.sqrt(norm1)
    phase=np.angle(wavevector)
    wavevector=wavevector/norm1
    
    reference = [f(X[i]) for i in range(2**m)]

    reference /= np.linalg.norm(reference)
    trial = phase/np.linalg.norm(phase)
    
    infidelity = 1 - trial.T@reference
    print('Infidelity=', infidelity)

    return trial, reference

def run_WSL_shot(f, m, e0, threshold=0, n_operators=20, shots=100000, method='decreasing_order', swap_option=True):
    q = QuantumRegister(m+1, 'q')
    circuit = QuantumCircuit(q)
    # classical_bits = ClassicalRegister(1)
    # circuit.add_register(classical_bits)
    qubits = [i for i in q]
    
    for i in range(m+1):  # Hadamard tower to initialize
        circuit.h(i)
        
    # Add the custom diagonal unitary
    diag_u, L_walsh_coeff = diagonal_unitary_1D(f,m,m,e0,threshold,n_operators,method,swap_option, input_walsh_coeff=None)

    ctrl_diag_u = diag_u.control(1)
    circuit.append(ctrl_diag_u, qubits)    
    
    # diag_u_dagger = diag_u.inverse()
    # ctrl_diag_u = diag_u_dagger.control(1, ctrl_state='0')
    # circuit.append(ctrl_diag_u, qubits) 

    circuit.h(0)
    circuit.s(0)
    #measurement in Walsh basis
    for i in range(1, m+1):
        circuit.h(i)
            
    print(circuit.draw())
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
    wavevector = np.zeros(2**m, dtype=float)
    for state, count in filtered_counts.items():
        state = state[len(state)//2:][::-1] + state[:len(state)//2][::-1]
        index = int(state, 2)
        wavevector[index] = np.sqrt(count / shots)
    # return print(wavevector)
    
    ref_operators = np.argsort(abs(np.array(L_walsh_coeff)))[-n_operators:]
    selected_operators = np.argsort(abs(np.array(wavevector)))[-n_operators:]
    print('reference_ops=', ref_operators,'\n','selected_ops=', selected_operators,'\n')
    if set(ref_operators) != set(selected_operators): print("Unexact set of operators, reduce e0 if necessary")  
    
    return run_WSL(f, m, e0, 0, n_operators, method, swap_option=True, input_walsh_coeff=wavevector)
