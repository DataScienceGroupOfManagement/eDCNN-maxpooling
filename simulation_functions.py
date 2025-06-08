import numpy as np

# function 1: low dimensional smooth radial function
def smooth_radial_func_noise_free_data(sample_size, data_dim):
    def y_func_d3(x):
        r_value = np.sqrt(np.sum(np.square(x))) * 2
        f_value = (r_value - 0.1) * (r_value - 0.5) * (r_value - 0.9)
        return f_value
    #function with dimension=4
    def y_func_d4(x):
        r_value = np.sqrt(np.sum(np.square(x))) * 2.2
        f_value = 0
        if r_value <= 1:
            f_value = pow(1 - r_value, 6) * (35 * pow(r_value, 2) + 18 * r_value + 3)
        elif r_value > 1:
            f_value = 0
        return f_value

    x_data = np.random.rand(sample_size, data_dim)
    x_data = (x_data - 0.5) * 1  # transform to [-1/2, 1/2]^d

    y_data = np.zeros((sample_size, 1))

    for index in range(sample_size):
        if data_dim == 3:
            y_data[index] = y_func_d3(x_data[index])
        elif data_dim == 4:
            y_data[index] = y_func_d4(x_data[index])
        else:
            y_data[index] = y_func_d4(x_data[index])

    return x_data, y_data

def smooth_radial_func_noisy_data(sample_size, data_dim, noise_variance=0.5):
    def y_func_d3(x):
        r_value = np.sqrt(np.sum(np.square(x))) * 2
        f_value = (r_value - 0.1) * (r_value - 0.5) * (r_value - 0.9)
        f_value += np.random.normal(scale=np.sqrt(noise_variance), )
        return f_value

    def y_func_d4(x):
        r_value = np.sqrt(np.sum(np.square(x))) * 2.2
        f_value = 0
        if r_value <= 1:
            f_value = pow(1 - r_value, 6) * (35 * pow(r_value, 2) + 18 * r_value + 3)
        elif r_value > 1:
            f_value = 0
        f_value += np.random.normal(scale=np.sqrt(noise_variance), )
        return f_value

    x_data = np.random.rand(sample_size, data_dim)
    x_data = (x_data - 0.5) * 1  # transform to [-1/2, 1/2]

    y_data = np.zeros((sample_size, 1))

    for index in range(sample_size):
        if data_dim == 3:
            y_data[index] = y_func_d3(x_data[index])
        elif data_dim == 4:
            y_data[index] = y_func_d4(x_data[index])
        else:
            y_data[index] = y_func_d4(x_data[index])

    return x_data, y_data

# function 2: a high-dimensional smooth function
def smooth_dim100_noise_free_data(sample_size, data_dim):
    if not data_dim == 100:
        raise ValueError

    def y_func_d100(x):
        f_value = 0
        for idx in range(data_dim):
            f_value +=  np.sin(x[idx] * np.pi / 2) **2
        f_value = np.exp(f_value / data_dim)
        return f_value

    x_data = np.random.rand(sample_size, data_dim) # x in [0,1]^d
    y_data = np.zeros((sample_size, 1))
    for index in range(sample_size):
        y_data[index] = y_func_d100(x_data[index])  # noise free data
    return x_data, y_data

# d=100, noisy data
def smooth_dim100_noisy_data(sample_size, data_dim, noise_variance):

    if not data_dim == 100:
        raise ValueError

    def y_func_d100(x):
        f_value = 0
        for idx in range(data_dim):
            f_value +=  np.sin(x[idx] * np.pi / 2) **2
        f_value = np.exp(f_value / data_dim)
        f_value += np.random.normal(scale=np.sqrt(noise_variance), )  # add noise
        return f_value

    x_data = np.random.rand(sample_size, data_dim)
    y_data = np.zeros((sample_size, 1))
    for index in range(sample_size):
        y_data[index] = y_func_d100(x_data[index])

    return x_data, y_data

