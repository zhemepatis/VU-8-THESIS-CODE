import numpy as np

def generate_vectors(dimension :int, domain_range, set_size :int):
    lower_end, higher_end = domain_range
    vectors = np.random.uniform(lower_end, higher_end, size = (set_size, dimension))
    return vectors

def generate_scalars(vectors, benchmark_function):
    values = np.array([benchmark_function(v) for v in vectors])
    return values
