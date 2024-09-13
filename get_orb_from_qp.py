"""
Created on Fri Jul 12 14:22:30 2024

@author: cesar
"""
file_aos_info_2 = '/home/cesar/qp2/src/real_space/He.ezfio.aos_info_2'
file_aos_info_3 = '/home/cesar/qp2/src/real_space/He.ezfio.aos_info_3'
file_mos_info = '/home/cesar/qp2/src/real_space/He.ezfio.mos_info'
#________ AO(x,y,z) = Sum_{k}^{n_gauss} ao_coef[k] * (x - ao_center[k])^{aj[k][0] * ... * exp(ao_alpha[k]*distance^2) }
class OrbitalSetup:
    """Setup config for ADAPT-VQE."""

    def __init__(self, file_aos_info_2, file_aos_info_3, file_mos_info) -> None:
        self.ao_center = []
        self.ao_n_gauss = []
        self.ao_axyz = []
        self.ao_coef = []
        self.ao_alpha = []
        self.mo_coef = []
        self.orbitals = {}  # Dictionary to store dynamically named functions
        self.mol_orbitals = {}
        self.which_grid = 'normal'
        if self.which_grid == 'normal':
            self.tg_0, self.tg_max = 0,1
            
        # Extract AO information from file_aos_info_2
        with open(file_aos_info_2, 'r') as f:
            for line in f:
                data = line.split()
                center_coords = list(map(float, data[:3]))
                exponants_polynome = list(map(int, data[3:6]))
                num_functions = int(data[6])

                self.ao_center.append(center_coords)
                self.ao_axyz.append(exponants_polynome)
                self.ao_n_gauss.append(num_functions)

        # Extract AO coefficients and coefficients exposants from file_aos_info_3
        with open(file_aos_info_3, 'r') as f:
            for line in f:
                data = line.split()
                index = len(self.ao_coef)
                num_functions = self.ao_n_gauss[index]
                coefficients = list(map(float, data[:num_functions]))
                coefficients_exposant = list(map(float, data[num_functions:num_functions * 2]))

                self.ao_coef.append(coefficients)
                self.ao_alpha.append(coefficients_exposant)


        # Extract MO coefficients from file_mos_info
        with open(file_mos_info, 'r') as f:
            for line in f:
                mo_coefficients = list(map(float, line.split()))
                self.mo_coef.append(mo_coefficients)
                
    