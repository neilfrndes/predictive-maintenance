import numpy as np

STATS = 'num_records, median, mean, std_dev, min_time, '\
        'max_time, quantile_10, quantile_90'


def calculate_stats(time_list):
    """Calculate mean and standard deviation of a list"""
    time_array = np.array(time_list)

    median = np.median(time_array)
    mean = np.mean(time_array)
    std_dev = np.std(time_array)
    max_time = np.amax(time_array)
    min_time = np.amin(time_array)
    quantile_10 = np.quantile(time_array, 0.1)
    quantile_90 = np.quantile(time_array, 0.9)

    return dict(median=median,
                mean=mean,
                std_dev=std_dev,
                min_time=min_time,
                max_time=max_time,
                quantile_10=quantile_10,
                quantile_90=quantile_90)
