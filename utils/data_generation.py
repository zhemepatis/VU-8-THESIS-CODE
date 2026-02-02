import numpy as np

def generate_vectors(dimentions, domain_range, set_size):
    lower_end, higher_end = domain_range
    vectors = np.random.uniform(lower_end, higher_end, size = (set_size, dimentions))
    return vectors

def generate_scalars(vectors, benchmark_function):
    values = np.array([benchmark_function(v) for v in vectors])
    return values
