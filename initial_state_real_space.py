#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:13:54 2024

@author: cesar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:30:11 2024

@author: cesar
"""
import numpy as np
# from scipy import integrate
# from scipy import sparse
# import scipy
# import matplotlib.pyplot as plt
# from matplotlib import animation
# from useful_functions import sparse_Mv
import math
import matplotlib.pyplot as plt
from walsh_jz import run_WSL, run_WSL_shot
from get_orb_from_qp import OrbitalSetup
from scipy.interpolate import griddata
from matplotlib import cm
file_aos_info_2 = '/home/cesar/qp2/src/real_space/He.ezfio.aos_info_2'
file_aos_info_3 = '/home/cesar/qp2/src/real_space/He.ezfio.aos_info_3'
file_mos_info = '/home/cesar/qp2/src/real_space/He.ezfio.mos_info'

setup = OrbitalSetup(file_aos_info_2, file_aos_info_3, file_mos_info)
axis = 'xy'
n_qubits = 8
n_walsh_operators = 20
e0 = 100  
shots = 1000000

class InitialStateGrid:
    def __init__(self, axis, n_qubits, n_walsh_operators, e0, shots, setup:OrbitalSetup):
        
        setup_attributes = [attr for attr in dir(setup) if '__' not in attr]
        for attr in setup_attributes:
            setattr(self, attr, setup.__dict__[attr])
            
        self.axis = axis
        self.dimension = len(self.axis)
        
        self.n_qubits = n_qubits
        N = 2**self.n_qubits
        assert(math.log(N,2) % self.dimension == 0)
        self.single_grid = np.linspace(-1, 1, int(N**(1/self.dimension)+0.1))
        self.target_grid = np.array(range(N))/N
            
        for k in range(len(self.ao_center)):
            self.create_atomic_orbital(k)
        
        for k in range(len(self.mo_coef)):
            self.create_mol_orbital(k)
        
        self.n_walsh_operators = n_walsh_operators
        self.e0 = e0
        self.shots = shots
        
    def create_atomic_orbital(self, indice):
        ao_center = self.ao_center[indice]
        ao_n_gauss = self.ao_n_gauss[indice]
        ao_axyz = self.ao_axyz[indice]
        ao_coef = self.ao_coef[indice]
        ao_alpha = self.ao_alpha[indice]

        def orbital(x):
            x,y,z = self.f_grid_custom(x)
            
            t = 0
            if x is None: x=ao_center[0]
            if y is None: y=ao_center[1]
            if z is None: z=ao_center[2]
            distance = (x - ao_center[0]) ** 2 + (y - ao_center[1]) ** 2 + (z - ao_center[2]) ** 2
            for k in range(ao_n_gauss):
                t += (ao_coef[k] *
                      (x - ao_center[0]) ** ao_axyz[0] *
                      (y - ao_center[1]) ** ao_axyz[1] *
                      (z - ao_center[2]) ** ao_axyz[2] *
                      np.exp(- ao_alpha[k] * distance))
            return t/100
        
        self.orbitals[f'orbital_{indice}'] = orbital
    
    def create_mol_orbital(self, indice):
        mo_coef = self.mo_coef[indice]
        assert(len(self.orbitals)==len(mo_coef))
        
        def mol_orbital(x):
            t=0
            for k in range(len(mo_coef)):
                t += mo_coef[k] * self.orbitals['orbital_'+str(k)](x)
            return t
        
        self.mol_orbitals[f'orbital_{indice}'] = mol_orbital

    def f_grid_custom(self, r):
        N = int(math.log(len(self.target_grid), 2))
        tol = 1e-6
        assert(r in self.target_grid)
        index = '0'*N + bin(int(np.where(np.abs(self.target_grid - r) <= tol)[0]))[2:]
        index = index[-N:]
        if self.dimension == 3:
            Nx, Ny, Nz = int(index[0:N//3],2), int(index[N//3:2*N//3],2), int(index[2*N//3:],2)
            x,y,z = self.single_grid[Nx], self.single_grid[Ny], self.single_grid[Nz]
            return x,y,z
        elif self.dimension == 2:
            if self.axis == 'xy':
                Nx, Ny = int(index[0:N//2],2), int(index[N//2:],2)
                x,y = self.single_grid[Nx], self.single_grid[Ny]
                return x,y,None
            elif self.axis == 'xz':
                Nx, Nz = int(index[0:N//2],2), int(index[N//2:],2)
                x,z = self.single_grid[Nx], self.single_grid[Nz]
                return x,None,z
            elif self.axis == 'yz':
                Ny, Nz = int(index[0:N//2],2), int(index[N//2:],2)
                y,z = self.single_grid[Ny], self.single_grid[Nz]
                return None,y,z
        else:
            if self.axis == 'x':
                Nx = int(index,2)
                return self.single_grid[Nx], None, None
            elif self.axis == 'y':
                Ny = int(index,2)
                return None, self.single_grid[Ny], None
            elif self.axis == 'z':
                Nz = int(index,2)
                return None, None, self.single_grid[Nz]
        
    def run_state_prep(self, orbital, n_qubits, e0=1, threshold=0, n_operators=10, shots=0, method='decreasing_order', swap_option=True):
        self.quantum_state, self.reference_state = run_WSL(orbital, n_qubits, e0, threshold, n_operators, method, swap_option)
        return
    
    def run_state_prep_shot(self, orbital, n_qubits, e0=1, threshold=0, n_operators=10, shots=100000, method='decreasing_order', swap_option=True):
        self.quantum_state, self.reference_state = run_WSL_shot(orbital, n_qubits, e0, threshold, n_operators, shots, method, swap_option)
        return
    
    def plot_1D_to_2D(self):
        self.quantum_state = np.array([self.quantum_state[i] * self.quantum_state[j] for i in range(len(self.quantum_state)) for j in range(len(self.quantum_state))])
        self.dimension = 2
        self.axis = 'xy'
        self.n_qubits = 2*self.n_qubits
        N = 2**self.n_qubits
        self.target_grid = np.array(range(N))/N
        return self.plot_state()
    
    def plot_state(self):
        position_list = [self.f_grid_custom(self.target_grid[i]) for i in range(len(self.target_grid))]
        # print(len(position_list))
        x, y, z = zip(*position_list)
        x, y, z = list(x), list(y), list(z)
        amplitude = self.quantum_state
        if self.dimension == 1:
                plt.figure(figsize=(5, 4))
                if self.axis == 'x': plt.plot(x, amplitude, color='blue', marker='o'), plt.plot(x, self.reference_state, color='green', label='reference')  # Line plot with markers                if self.axis == 'x': plt.plot(x, amplitude, color='blue')  # Line plot with markers
                elif self.axis == 'y': plt.plot(y, amplitude, color='blue', marker = 'o'), plt.plot(y, self.reference_state, color='green', label='reference')  # Line plot with markers
                elif self.axis == 'z': plt.plot(z, amplitude, color='blue', marker = 'o'), plt.plot(z, self.reference_state, color='green', label='reference')  # Line plot with markers
                plt.xlabel(self.axis)
                plt.ylabel('Amplitude')
                plt.title('Quantum State Preparation, 1D orbital')
                
        elif self.dimension == 2:  
                if self.axis == 'xy': x, y = np.array(x), np.array(y)
                elif self.axis == 'xz': x, y = np.array(x), np.array(z)
                elif self.axis == 'yz': x, y = np.array(y), np.array(z)
                xi = np.linspace(min(x), max(x), len(self.single_grid))
                yi = np.linspace(min(y), max(y), len(self.single_grid))
                xi, yi = np.meshgrid(xi, yi)
                
                Z = griddata((x, y), amplitude, (xi, yi), method='cubic')
                #3D surface plot
                fig = plt.figure(figsize=(5, 4))
                ax = fig.add_subplot(111, projection='3d')
                surf = ax.plot_surface(xi, yi, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

                cb = fig.colorbar(surf)
                cb.set_label('Amplitude')
                ax.set_xlabel(self.axis[0])
                ax.set_ylabel(self.axis[1])
                # ax.set_zlabel('Amplitude')
                plt.title('Quantum State Preparation, 3D surface plot with amplitude')
                plt.show()
                #____________________________________________________________________________________________
                #2D isosurface plot
                k=len(set(amplitude))
                if not np.isnan(amplitude).any():
                    amplitudes = sorted(set(amplitude), reverse=True)[:k][::-1]
                    fig = plt.figure(figsize=(5, 4))
                    cmap = plt.get_cmap('seismic')  
                    fig, ax = plt.subplots(figsize=(5, 4))
                    contour = ax.contourf(xi, yi, Z, levels=amplitudes, cmap=cmap)
                    fig.colorbar(contour, ax=ax, orientation='vertical')
                    ax.set_xlabel(self.axis[0])
                    ax.set_ylabel(self.axis[1])
                    plt.show()
                
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(x, y, z, c=amplitude, cmap='viridis')
            cb = plt.colorbar(sc)
            cb.set_label('Amplitude')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.title('3D Scatter Plot with Amplitude')
            plt.show()
    
    
    
m = InitialStateGrid(axis, n_qubits, n_walsh_operators, e0, shots, setup)
# for orbital_name in m.orbitals.keys():
#     m.run_state_prep(m.orbitals[orbital_name], m.n_qubits)
#     m.plot_state()
#     m.run_state_prep_shot(m.orbitals[orbital_name], m.n_qubits)

#     m.plot_state()
#     break

m.run_state_prep(m.orbitals['orbital_4'], m.n_qubits, m.e0, 0, m.n_walsh_operators, m.shots, method='decreasing_order', swap_option=True)
m.plot_state()
# print('_____________________NOW WITH SHOTS___________________')
m.run_state_prep_shot(m.orbitals['orbital_4'], m.n_qubits, m.e0, 0, m.n_walsh_operators, m.shots, method='decreasing_order', swap_option=True)

m.plot_state()
print('\nTotal infidelity =', 1 - np.abs(np.dot(np.conjugate(m.quantum_state),m.reference_state))**2)




