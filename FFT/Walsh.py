import os, h5py 
import numpy as np
import multiprocessing as mp 

import UsefulFunctions as uf 

class WalshConstructor:

    def __init__(self, n_qubits, n_operators, n_cpus, dir_path, f):

        self.n_qubits = n_qubits 
        self.n_operators = n_operators 
        self.N = 2 ** n_qubits 

        self.n_cpus = n_cpus 

        self.dir_path = dir_path

        self.f = f

    def parallel_sparse_walsh_series(self):

        ##Paths and links to files setup: DO NOT TOUCH IF NOT MANDATORY
        master_path = uf.path_to_master_file(self.dir_path, self.N)
        uf.ExternalLinkSetup(self.dir_path, master_path, self.N)

        L = self.parallel_list_walsh_coeff(self.dir_path)

        degrees = np.argsort(abs(np.array(L)))[-self.n_operators:]
        coeffs = [L[i] for i in degrees]

        #Using the Walsh database file
        data_file = h5py.File(master_path, 'r')    

        #Main function
        results = self.parallel_walsh_reconstruction(
            degrees, 
            coeffs, 
            master_path, 
        )
        
        return results

    def parallel_walsh_reconstruction(self, degrees, coefficients, master_path):

        if len(degrees) != len(coefficients):
            raise ValueError("Degrees and coefficients must have the same length.")
        
        data_file = h5py.File(master_path, 'r')

        walsh_coefficients = []
        for degree in degrees:
            file_index = uf.select_correct_file(degree, self.N, self.dir_path)
            walsh_coefficients.append(list(data_file[f'/data{file_index}'][f'walsh_coeff_for_{degree}']))
            
        list_N = list(uf.chunk_function(range(self.N), self.n_cpus))

        global final_reconstruction
        def final_reconstruction(index, list_N, list_coeffs, coeffs):

            temp_results = []
            for i in list_N:
                w_value = 0
                for i_c in range(len(coeffs)):
                    w_value += coeffs[i_c] * list_coeffs[i_c][i]
                temp_results.append(w_value)
            
            return (index, temp_results)
        
        with mp.Pool(processes=self.n_cpus) as pool:
            pool_results = pool.starmap(final_reconstruction, [(i, list_N[i], walsh_coefficients, coefficients) for i in range(self.n_cpus)])

        final_results = []
        for i in range(len(pool_results)):
            final_results += pool_results[i][1]
        del pool_results
        
        return final_results

    def parallel_list_walsh_coeff(self, path):

        list_f_x = uf.list_f(self.f, self.N)

        list_N = list(uf.chunk_function(range(self.N), self.n_cpus))

        global one_list_walsh_coeff
        def one_list_walsh_coeff(index, list_f, list_N, path):

            res = []
            for j in list_N:
                a = uf.walsh_coeff_from_file(j, path, list_f, self.N)
                if abs(a) < 1e-10:
                    res.append(0)
                else:
                    res.append(a)
            
            return (index, res)
        
        with mp.Pool(processes=self.n_cpus) as pool:
            pool_results = pool.starmap(one_list_walsh_coeff, [(i, list_f_x, list_N[i], path) for i in range(self.n_cpus)])
        
        result = []
        for i in range(self.n_cpus):
            result += pool_results[i][1]
        
        return result