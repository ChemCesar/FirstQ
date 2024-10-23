def run(qubits: int, cpus:int):

    import os
    import numpy as np 
    from timeit import default_timer as timer
    from dyadic_gen import DyadicGenerator

    start = timer()

    script_dir = os.path.dirname(__file__)
    generator = DyadicGenerator(qubits, cpus, script_dir)

    generator.parallel_walsh_coeff_gen_file()

    end = timer() - start 

    print(f"Runtime to generate the database for {qubits} qubits: {end} s")

if __name__ == "__main__":

    import argparse 

    parser = argparse.ArgumentParser(
        epilog="Dyadic coefficients database generator",
        usage="python database_generator.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-n",
        "--qubits",
        help="Number of qubits",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-c",
        "--cpus",
        help="Number of cpus",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    run(args.qubits, args.cpus)