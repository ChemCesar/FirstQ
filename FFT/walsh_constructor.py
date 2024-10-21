def run(n_qubits: int, n_operators: int, n_cpus: int):

    import os
    import numpy as np 
    from timeit import default_timer as timer
    import matplotlib.pyplot as plt

    from Walsh import WalshConstructor

    def f(x):
        
        return np.cos(2 * np.pi * x)

    start = timer()

    #Additional setup, DO NOT TOUCH IF NOT MANDATORY
    script_dir = os.path.dirname(__file__)
    N = 2 ** n_qubits

    #Generation of the results
    walsh = WalshConstructor(n_qubits, n_operators, n_cpus, script_dir, f)
    result = walsh.parallel_sparse_walsh_series()

    x = np.linspace(0, 1, N)
    plt.plot(x, result)
    plt.plot(x, f(x))
    plt.savefig("Example.eps")

    end = timer() - start 
    print(f"Total Runtime for generating the Walsh decomposition of the function f : {end} s")

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        epilog="Walsh approximation generator",
        usage="python walsh_constructor.py",
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
    parser.add_argument(
        "-o",
        "--operators",
        help="Number of operators",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    run(args.qubits, args.operators, args.cpus)