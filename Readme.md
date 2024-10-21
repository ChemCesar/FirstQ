# Walsh generator

## Contents 

You have two main files here, one for generating the dataset files and one to use the dataset to construct the walsh series for a given function.

## How to use the walsh_constructor.py code

The ```walsh_constructor.py``` script is the main file to construct the walsh series when the dataset is stored on the computer. It can be used as follows:

```
python walsh_constructor.py -n $NUM_QUBITS -c $NUM_CPUS -o $NUM_OPS
```

where: 
- $NUM_QUBITS is the number of qubits (current stored database is limited at 16qubits because of datasize)
- $NUM_CPUS is the number of CPUs you want to use for the computation
- $NUM_OPS is the number of operators you want for the Walsh series

Additionnally, in this file you only have to change the function you want to approximate with a Walsh series. This can be done in the ```run()``` part of the script, by modifying the definition of the function ```f()```.

Remark:
Do not modify the rest of the inputs in this file, especially the ```script_dir``` variable, which is mandatory to find the path in order to recover the database.

## How to use the database_generator.py

The ```database_generator.py``` script is the main file to generate the database. It can be used as follows:

```
python database_generator.py -n $NUM_QUBITS -c $NUM_CPUS
```

where:
- $NUM_QUBITS is the number of qubits you want to generate the database for (notice that above 18qubits the storage will need more than 1To!)
- $NUM_CPUS is the number of cpus you want to use for generating the database for this particular number of qubits.

Note that below 10qubits, it is less efficient to use several CPUS compared to one.