import numpy as np
import h5py
import os, sys 
import multiprocessing as mp

import UsefulFunctions as uf
    
class DyadicGenerator:

    def __init__(self, n_qubits, n_cpus, dir_path) -> None:

        self.n_qubits = n_qubits
        self.n_cpus = n_cpus 
        self.N = 2 ** n_qubits

        self.dir_path = dir_path

        self.database_path = uf.path_to_database(self.dir_path, self.N)
        self.master_path = uf.path_to_master_file(self.dir_path, self.N)
    
        self.creating_data_directories()
    
    def parallel_walsh_coeff_gen_file(self):

        list_N = list(uf.chunk_function(list(range(self.N)), self.n_cpus))

        global parallel_walsh_coeff_gen
        def parallel_walsh_coeff_gen(index, list_N, local_path, N):
        
            temp_path = os.path.join(self.database_path, f'part{index + 1}.hdf5') 
            with h5py.File(temp_path, 'w') as file:
                group = file.create_group("/path/to/data")    
                for i in list_N:
                    group.create_dataset(f'walsh_coeff_for_{i}', data=uf.walsh_coeff(i, N))
            
            #return (index, res) 

        with mp.Pool(processes=self.n_cpus) as pool:
            pool_results = pool.starmap(parallel_walsh_coeff_gen, [(i, list_N[i], self.database_path, self.N) for i in range(self.n_cpus)])
        
        with h5py.File(self.master_path, 'w') as file:
            for i in range(self.n_cpus):
                file[f'/data{i+1}'] = h5py.ExternalLink(os.path.join(self.database_path, f'part{i + 1}.hdf5'), '/path/to/data')
    
    def creating_data_directories(self):

        database_directory = os.path.join(self.dir_path, 'database/')
        qubits_directory = os.path.join(database_directory, f'{self.n_qubits}qubits/')

        try:
            os.mkdir(database_directory)
        except FileExistsError:
            print(f"Directory {database_directory} already exists")

        try:
            os.mkdir(qubits_directory)
        except FileExistsError:
            print(f"Directory {qubits_directory} already exists")
        
