#!/usr/bin/env python
# coding: utf-8
import os
# # Dask client setup

# In[1]:


# from dask.distributed import LocalCluster
#
# cluster = LocalCluster(n_workers=4, processes=True,
#                        threads_per_worker=1)  # Fully-featured local Dask cluster
# client = cluster.get_client()
# client

# In[2]:


import sys

sys.path.append('../')

import algorithms.lloyd_clustering as lloyd

# # Generate data

# In[18]:


from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np

# Generate data for different center amounts
for center_amount in range(2, 11):
    X, y = make_blobs(n_samples=1000, centers=center_amount, n_features=2, random_state=42)
    X = StandardScaler().fit_transform(X)
    with open(f"{os.environ['VSC_DATA']}/data/data_center_{center_amount}.txt", 'w') as file:
        file.write(str([tuple(x) for x in X.tolist()]))

# Generate data for different dimension amounts
for dimension_amount in range(2, 11):
    X, y = make_blobs(n_samples=1000, centers=5, n_features=dimension_amount, random_state=42)
    X = StandardScaler().fit_transform(X)
    with open(f"{os.environ['VSC_DATA']}/data/data_dimension_{dimension_amount}.txt", 'w') as file:
        file.write(str([tuple(x) for x in X.tolist()]))

# Generate data for different sample amounts
for sample_amount in range(1000, 10001, 1000):
    X, y = make_blobs(n_samples=sample_amount, centers=5, n_features=2, random_state=42)
    X = StandardScaler().fit_transform(X)
    with open(f"{os.environ['VSC_DATA']}/data/data_sample_{sample_amount}.txt", 'w') as file:
        file.write(str([tuple(x) for x in X.tolist()]))

# ## Test the dask version with the dashboard to see if the code is running in parallel

# In[3]:


# lloyd.lloyd_algorithm('../data/data_05.txt", 5, True, 3)

# it's visibly parallel but the data is quite small

# # Run all versions of the algorithm.
# Use memray and timeit to measure performance.

# ## Import all required extra packages for benchmarking

# In[4]:


import logging
import timeit
from memory_profiler import memory_usage
import ast
import dask.array as da
import csv


# In[2]:


def write_benchmark_to_csv(file: str, data: list):
    with open(file, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(data)


# In[3]:


def process_data(data: list):
    mean_val = np.mean(data)
    min_val = np.min(data)
    max_val = np.max(data)
    std_deviation = np.std(data)
    return mean_val, min_val, max_val, std_deviation


# In[1]:


def prepare_files(files: list[str]):
    for output in files:
        with open(output, 'w') as outputfile:
            writer = csv.writer(outputfile)
            writer.writerow(['amount', 'mean', 'min', 'max', 'std_deviation'])


# ## Time benchmarks using timeit.

# In[5]:

logging.basicConfig(filename=f'{os.environ["VSC_DATA"]}/output/benchmark.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

log = logging.getLogger('lloyd_benchmark')


def time_benchmark():
    log.info('Starting time benchmark')
    paths = [
        f"{os.environ['VSC_DATA']}/output/output_time_center_base.csv",
        f"{os.environ['VSC_DATA']}/output/output_time_center_numpy.csv",
        f"{os.environ['VSC_DATA']}/output/output_time_center_dask.csv",
        f"{os.environ['VSC_DATA']}/output/output_time_dimension_base.csv",
        f"{os.environ['VSC_DATA']}/output/output_time_dimension_numpy.csv",
        f"{os.environ['VSC_DATA']}/output/output_time_dimension_dask.csv",
        f"{os.environ['VSC_DATA']}/output/output_time_sample_base.csv",
        f"{os.environ['VSC_DATA']}/output/output_time_sample_numpy.csv",
        f"{os.environ['VSC_DATA']}/output/output_time_sample_dask.csv"
    ]

    prepare_files(paths)

    for center_amount in range(2, 11):
        # Read the data from the file
        with open(f"{os.environ['VSC_DATA']}/data/data_center_{center_amount}.txt", encoding="utf-8") as file:
            filecontent = file.read()
            points: list[tuple[float, ...]] = list(ast.literal_eval(filecontent))

        # Initialize lists to store the times
        base_times = []
        numpy_times = []
        dask_times = []

        # Iterate 100 times to check means and error margin.
        for i in range(0, 100):
            initial_centers = lloyd.k_means_plus_plus(points, center_amount)
            # Time the implementations
            base_times.append(timeit.timeit(lambda: lloyd.k_means_base(points, initial_centers), number=1))
            numpy_times.append(timeit.timeit(lambda: lloyd.k_means_numpy(np.array(points), np.array(initial_centers)), number=1))
            dask_times.append(timeit.timeit(lambda: lloyd.k_means_dask(da.from_array(points, chunks=(len(points) // 4, len(points[0]))), np.array(initial_centers)), number=1))

            log.info(f"saving results for center amount: {center_amount}, iteration: {i}")
        # Save the results to the file
        result_base = [center_amount, *process_data(base_times)]
        result_numpy = [center_amount, *process_data(numpy_times)]
        result_dask = [center_amount, *process_data(dask_times)]
        write_benchmark_to_csv(f"{os.environ['VSC_DATA']}/output/output_time_center_base.csv", result_base)
        write_benchmark_to_csv(f"{os.environ['VSC_DATA']}/output/output_time_center_numpy.csv", result_numpy)
        write_benchmark_to_csv(f"{os.environ['VSC_DATA']}/output/output_time_center_dask.csv", result_dask)

    log.info('Finished center amount benchmark')

    for dimensions in range(2, 11):
        #Read the data from the file
        with open(f"{os.environ['VSC_DATA']}/data/data_dimension_{dimensions}.txt", encoding="utf-8") as file:
            filecontent = file.read()
            points: list[tuple[float, ...]] = list(ast.literal_eval(filecontent))

        # Initialize lists to store the times
        base_times = []
        numpy_times = []
        dask_times = []

        # Iterate 100 times to check means and error margin.
        for i in range(0, 100):
            initial_centers = lloyd.k_means_plus_plus(points, 5)

            # Time the implementations
            base_times.append(timeit.timeit(lambda: lloyd.k_means_base(points, initial_centers), number=1))
            numpy_times.append(timeit.timeit(lambda: lloyd.k_means_numpy(np.array(points), np.array(initial_centers)), number=1))
            dask_times.append(timeit.timeit(lambda: lloyd.k_means_dask(da.from_array(points, chunks=(len(points) // 4, len(points[0]))), np.array(initial_centers)), number=1))

            log.info(f"saving results for dimension amount: {dimensions}, iteration: {i}")
        # Save the results to the file
        result_base = [dimensions, *process_data(base_times)]
        result_numpy = [dimensions, *process_data(numpy_times)]
        result_dask = [dimensions, *process_data(dask_times)]
        write_benchmark_to_csv(f"{os.environ['VSC_DATA']}/output/output_time_dimension_base.csv", result_base)
        write_benchmark_to_csv(f"{os.environ['VSC_DATA']}/output/output_time_dimension_numpy.csv", result_numpy)
        write_benchmark_to_csv(f"{os.environ['VSC_DATA']}/output/output_time_dimension_dask.csv", result_dask)

    log.info('Finished dimension amount benchmark')

    for sample_amount in range(1000, 10001, 1000):
        # Read the data from the file
        with open(f"{os.environ['VSC_DATA']}/data/data_sample_{sample_amount}.txt", encoding="utf-8") as file:
            filecontent = file.read()
            points: list[tuple[float, ...]] = list(ast.literal_eval(filecontent))

        # Initialize lists to store the times
        base_times = []
        numpy_times = []
        dask_times = []

        # Iterate 100 times to check means and error margin.
        for i in range(0, 100):
            initial_centers = lloyd.k_means_plus_plus(points, 5)

            # Time the implementations
            base_times.append(timeit.timeit(lambda: lloyd.k_means_base(points, initial_centers), number=1))
            numpy_times.append(timeit.timeit(lambda: lloyd.k_means_numpy(np.array(points), np.array(initial_centers)), number=1))
            dask_times.append(timeit.timeit(lambda: lloyd.k_means_dask(da.from_array(points, chunks=(len(points) // 4, len(points[0]))), np.array(initial_centers)), number=1))

            log.info(f"saving results for sample amount: {sample_amount}, iteration: {i}")
        # Save the results to the file
        result_base = [sample_amount, *process_data(base_times)]
        result_numpy = [sample_amount, *process_data(numpy_times)]
        result_dask = [sample_amount, *process_data(dask_times)]
        write_benchmark_to_csv(f"{os.environ['VSC_DATA']}/output/output_time_sample_base.csv", result_base)
        write_benchmark_to_csv(f"{os.environ['VSC_DATA']}/output/output_time_sample_numpy.csv", result_numpy)
        write_benchmark_to_csv(f"{os.environ['VSC_DATA']}/output/output_time_sample_dask.csv", result_dask)
    log.info('Finished sample amount benchmark')

    log.info('Finished time benchmark')


# # Memory benchmarks using memory_profiler

# In[ ]:


def mem_benchmark():
    log.info('Starting memory benchmark')
    paths = [
        f"{os.environ['VSC_DATA']}/output/output_mem_center_base.csv",
        f"{os.environ['VSC_DATA']}/output/output_mem_center_numpy.csv",
        f"{os.environ['VSC_DATA']}/output/output_mem_center_dask.csv",
        f"{os.environ['VSC_DATA']}/output/output_mem_dimension_base.csv",
        f"{os.environ['VSC_DATA']}/output/output_mem_dimension_numpy.csv",
        f"{os.environ['VSC_DATA']}/output/output_mem_dimension_dask.csv",
        f"{os.environ['VSC_DATA']}/output/output_mem_sample_base.csv",
        f"{os.environ['VSC_DATA']}/output/output_mem_sample_numpy.csv",
        f"{os.environ['VSC_DATA']}/output/output_mem_sample_dask.csv"
    ]

    prepare_files(paths)

    for center_amount in range(2, 11):
        # Read the data from the file
        with open(f"{os.environ['VSC_DATA']}/data/data_center_{center_amount}.txt", encoding="utf-8") as file:
            filecontent = file.read()
            points: list[tuple[float, ...]] = list(ast.literal_eval(filecontent))

        # Initialize lists to store the memory usage
        base_mem = ()
        numpy_mem = ()
        dask_mem = ()

        # Iterate 100 times to check means and error margin.
        for i in range(0, 100):
            initial_centers = lloyd.k_means_plus_plus(points, center_amount)

            # Memory the implementations
            base_mem.append(memory_usage((lloyd.k_means_base, (points, initial_centers))))
            numpy_mem.append(memory_usage((lloyd.k_means_numpy, (np.array(points), initial_centers))))
            dask_mem.append(memory_usage((lloyd.k_means_dask, (da.array(points), initial_centers)), multiprocess=True))

            log.info(f"saving results for center amount: {center_amount}, iteration: {i}")
        # Save the results to the file
        result_base = [center_amount, *process_data(base_mem)]
        result_numpy = [center_amount, *process_data(numpy_mem)]
        result_dask = [center_amount, *process_data(dask_mem)]
        write_benchmark_to_csv(f"{os.environ['VSC_DATA']}/output/output_mem_center_base.csv", result_base)
        write_benchmark_to_csv(f"{os.environ['VSC_DATA']}/output/output_mem_center_numpy.csv", result_numpy)
        write_benchmark_to_csv(f"{os.environ['VSC_DATA']}/output/output_mem_center_dask.csv", result_dask)

    log.info('Finished center amount benchmark')

    for dimensions in range(2, 11):
        # Read the data from the file
        with open(f"{os.environ['VSC_DATA']}/data/data_dimension_{dimensions}.txt", encoding="utf-8") as file:
            filecontent = file.read()
            points: list[tuple[float, ...]] = list(ast.literal_eval(filecontent))

        # Initialize lists to store the memory usage
        base_mem = ()
        numpy_mem = ()
        dask_mem = ()

        # Iterate 100 times to check means and error margin.
        for i in range(0, 100):
            initial_centers = lloyd.k_means_plus_plus(points, 5)

            # Memory the implementations
            base_mem.append(memory_usage((lloyd.k_means_base, (points, initial_centers))))
            numpy_mem.append(memory_usage((lloyd.k_means_numpy, (np.array(points), initial_centers))))
            dask_mem.append(memory_usage((lloyd.k_means_dask, (da.array(points), initial_centers)), multiprocess=True))

            log.info(f"saving results for dimension amount: {dimensions}, iteration: {i}")
        # Save the results to the file
        result_base = [dimensions, *process_data(base_mem)]
        result_numpy = [dimensions, *process_data(numpy_mem)]
        result_dask = [dimensions, *process_data(dask_mem)]
        write_benchmark_to_csv(f"{os.environ['VSC_DATA']}/output/output_mem_dimension_base.csv", result_base)
        write_benchmark_to_csv(f"{os.environ['VSC_DATA']}/output/output_mem_dimension_numpy.csv", result_numpy)
        write_benchmark_to_csv(f"{os.environ['VSC_DATA']}/output/output_mem_dimension_dask.csv", result_dask)

    log.info('Finished dimension amount benchmark')

    for sample_amount in range(1000, 10001, 1000):
        # Read the data from the file
        with open(f"{os.environ['VSC_DATA']}/data/data_sample_{sample_amount}.txt", encoding="utf-8") as file:
            filecontent = file.read()
            points: list[tuple[float, ...]] = list(ast.literal_eval(filecontent))

        # Initialize lists to store the memory usage
        base_mem = ()
        numpy_mem = ()
        dask_mem = ()

        # Iterate 100 times to check means and error margin.
        for i in range(0, 100):
            initial_centers = lloyd.k_means_plus_plus(points, 5)

            # Memory the implementations
            base_mem.append(memory_usage((lloyd.k_means_base, (points, initial_centers))))
            numpy_mem.append(memory_usage((lloyd.k_means_numpy, (np.array(points), initial_centers))))
            dask_mem.append(memory_usage((lloyd.k_means_dask, (da.array(points), initial_centers)), multiprocess=True))

            log.info(f"saving results for sample amount: {sample_amount}, iteration: {i}")
        # Save the results to the file
        result_base = [sample_amount, *process_data(base_mem)]
        result_numpy = [sample_amount, *process_data(numpy_mem)]
        result_dask = [sample_amount, *process_data(dask_mem)]
        write_benchmark_to_csv(f"{os.environ['VSC_DATA']}/output/output_mem_sample_base.csv", result_base)
        write_benchmark_to_csv(f"{os.environ['VSC_DATA']}/output/output_mem_sample_numpy.csv", result_numpy)
        write_benchmark_to_csv(f"{os.environ['VSC_DATA']}/output/output_mem_sample_dask.csv", result_dask)

    log.info('Finished sample amount benchmark')

    log.info('Finished memory benchmark')


# # Run the benchmarks

# In[ ]:


time_benchmark()
mem_benchmark()

# # clean up environment

# In[ ]:


# client.close()
# cluster.close()

# # The results are in the output folder
