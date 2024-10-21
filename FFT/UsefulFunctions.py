import os, sys
import h5py 
import numpy as np 
from collections import deque 

##Walsh technical functions##
def dyadic(x, n):

    inv_pow_two = 1 / (2 ** np.arange(1, n+1))
    L = np.zeros(n)

    a = x 
    for i in range(n):
        if a >= inv_pow_two[i]:
            L[i] = 1
            a -= inv_pow_two[i]
    
    return L 

def walsh(j,x):

    j_bits = np.array(list(f"{j:b}")[::-1], dtype=int)

    X = dyadic(x, len(j_bits))

    p = np.dot(j_bits, X)

    return (-1)**p 

def walsh_coeff(j, N):

    X = np.linspace(0, 1, N)

    res = []
    for i in range(N):
        res.append(walsh(j, X[i]))
    
    return res

def list_f(f, N):

    X = np.linspace(0, 1, N)
    list_f = [f(e) for e in X]

    return list_f

##MP related functions##
def chunk_function(list_N, num_jobs):

    deque_obj = deque(list_N)
    chunk_size = int(len(list_N)/num_jobs) + 1

    while deque_obj:
        chunk= []
        for _ in range(chunk_size):
            if deque_obj:
                chunk.append(deque_obj.popleft())
        yield chunk

def chunk_walsh_coeffs(walsh_coefficients, num_jobs, num_degrees):

    temp_walsh = []
    for i in range(num_degrees):
        temp_walsh.append(list(chunk_function(walsh_coefficients[i], num_jobs)))
    
    final_walsh = []
    for i in range(num_jobs):
        temp_list = []
        for j in range(num_degrees):
            temp_list.append(temp_walsh[j][i])
        final_walsh.append(temp_list)
        
    return final_walsh

##Files related functions##
def ExternalLinkSetup(dir_path, path, N):

    database_path = path_to_database(dir_path, N)
    num_cpus = counting_number_files_in_rep(database_path)

    with h5py.File(path, 'w') as file:
        for i in range(num_cpus):
            local_path = os.path.join(database_path, f'part{i+1}.hdf5')
            file[f'/data{i+1}'] = h5py.ExternalLink(local_path, 'path/to/data')  

def counting_number_files_in_rep(dir_path):

    import fnmatch 

    count = len(fnmatch.filter(os.listdir(dir_path), '*.hdf5*'))
    return count -1 

def select_correct_file(j, N, dir_path):

    path = path_to_database(dir_path, N)

    num_files = counting_number_files_in_rep(path)

    list_j = list(chunk_function(range(N), num_files))

    for i in range(len(list_j)):
        if j in list_j[i]:
            return i+1

def path_to_database(dir_path, N):

    n = int(np.log(N)/np.log(2))

    path = os.path.join(dir_path, f'database/{n}qubits/')
    # path = os.path.join(dir_path, 'temp/')

    return path

def path_to_master_file(dir_path, N):

    database_path = path_to_database(dir_path, N)
    path = os.path.join(database_path, 'master.hdf5')
    # path = os.path.join(dir_path, 'testfile.hdf5')

    return path

def walsh_coeff_from_file(j, path, list_f, N):

    index_file = select_correct_file(j, N, path)

    file_path = path_to_master_file(path, N)

    data_file = h5py.File(file_path, 'r')
    dset = data_file[f'/data{index_file}'][f'walsh_coeff_for_{j}']
    data = dset[:]

    a = 0
    for i in range(N):
        a += list_f[i] * data[i] / N
    
    return a

# def walsh_coeff_gen_file(N, path):

#     f = h5py.File(path, 'w')

#     for j in range(N):
#         res = walsh_coeff(j, N)
#         f.create_dataset(f'walsh_coeff_for_{j}', data=res)
