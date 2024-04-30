# https://dodona.be/nl/courses/3363/series/36091/activities/207273748/#
# Date: 26/04/24
# Description: Lloyd's clustering algorithm module
import ast
import importlib.util
import random
import timeit
from math import sqrt

import numpy as np


def lloyd_algorithm(data: str, k: int, k_plus_plus_init: bool = False, variant: int = 3) -> set[tuple[float, ...]]:
    """
    Implementation of the lloyd clustering algorithm for solving the k-means problem. We're using 3 versions of the
    k-means algorithm:
    - base implementation: The basic implementation of the lloyd clustering algorithm.
    - numpy implementation: The numpy implementation for improved performance on larger datasets.
    - dask implementation: The  dask implementation for parallellization to improve execution time.
    :param variant: 1 -> base implementation, 2 -> numpy implementation, 3 -> dask implementation
    :param data: Path to the file with the data.
    :param k: Amount of clusters wanted.
    :param k_plus_plus_init: Boolean to indicate if the k-means++ initialization should be used or the first k points.
    :return: A set of tuples with the coordinates of the centroids.
    >>> lloyd_algorithm("../data/data02.txt", 2, False,1)
    {(85.8, 46.4, 31.0), (46.2, 46.46666666666667, 51.06666666666667)}
    >>> lloyd_algorithm("../data/data02.txt", 2, False,2)
    {(np.float64(85.8), np.float64(46.4), np.float64(31.0)), (np.float64(46.2), np.float64(46.46666666666667), np.float64(51.06666666666667))}
    >>> lloyd_algorithm("../data/data02.txt", 2, False,3)
    {(np.float64(85.8), np.float64(46.4), np.float64(31.0)), (np.float64(46.2), np.float64(46.46666666666667), np.float64(51.06666666666667))}
    """

    # Read the data from the file
    with open(data, encoding="utf-8") as file:
        filecontent = file.read()
        points: list[tuple[float, ...]] = list(ast.literal_eval(filecontent))

    # Initialize the centroids
    if k_plus_plus_init:
        initial_centers = k_means_plus_plus(points, k)
    else:
        initial_centers = points[:k]

    # Run the desired variant of the k-means algorithm
    if variant == 1:
        return k_means_base(points, initial_centers)
    if importlib.util.find_spec("dask") is None or variant == 2:
        return k_means_numpy(np.array(points), np.array(initial_centers))
    return k_means_dask(da.from_array(points, chunks=(len(points) // 4, len(points[0]))), np.array(initial_centers))


def k_means_plus_plus(data: list[tuple[float, ...]], k: int) -> list[tuple[float, ...]]:
    """
    Implementation of the k-means++ initialization.
    :param data: List of tuples with the coordinates of the points.
    :param k: Amount of clusters wanted.
    :return: A set of tuples with the coordinates of the centroids.
    Can't really test this function because it's random.
    >>> centroids = k_means_plus_plus([(1, 2), (3, 4), (5, 6), (7, 8)], 2)
    >>> isinstance(centroids, list)
    True
    >>> len(centroids) == 2
    True
    """

    # Select the first center randomly
    centers = [random.choice(data)]

    while len(centers) < k:
        # Calculate the distance to the closest center for each point
        distances = [min(_euclid(point, center) for center in centers) for point in data]

        # Calculate the probability of each point to be the next center
        probabilities = [distance ** 2 for distance in distances]
        sum_probabilities = sum(probabilities)
        probabilities = [probability / sum_probabilities for probability in probabilities]

        # Select the next center randomly
        centers.append(random.choices(data, probabilities)[0])

    return centers


# BASE IMPLEMENTATION
def k_means_base(data: list[tuple[float, ...]], initial_centers: list[tuple[float, ...]]) -> set[tuple[float, ...]]:
    """
    Implementation of the k-means algorithm. This uses a very basic implementation of the algorithm with own code for
    base functionality.
    :param data: List of tuples with the coordinates of the points.
    :param initial_centers: List of tuples with the coordinates of the initial centroids.
    :return: A set of tuples with the coordinates of the centroids.
    >>> k_means_base([(1, 2), (3, 4), (5, 6), (7, 8)], [(1, 2), (3, 4)])
    {(2.0, 3.0), (6.0, 7.0)}

    This also works for more than 2 dimensions.
    >>> k_means_base([(1, 2, 3), (4, 5, 6), (7, 8, 9)], [(1, 2, 3), (4, 5, 6)])
    {(5.5, 6.5, 7.5), (1.0, 2.0, 3.0)}
    """

    prev_centers: list[tuple[float, ...]] = []
    centers = initial_centers
    while prev_centers != centers:
        # Centers to clusters

        clusters: dict[tuple[float, ...], list[tuple[float, ...]]] = {center: [] for center in centers}
        for point in data:
            # Get the closest center for each point.
            closest_center = _get_closest_center(point, centers)
            clusters[closest_center].append(point)

        # Clusters to centers
        new_centers = []
        for center, points in clusters.items():
            # Calculate the new center for each cluster with the gravity of gravity.
            new_center = tuple(
                (sum(point[i] for point in points) / len(points)) for i in range(len(points[0]))
            )
            new_centers.append(new_center)

        prev_centers = centers
        centers = new_centers

    return set(centers)


def _euclid(point: tuple[float, ...], center: tuple[float, ...]) -> float:
    """
    Calculate the Euclidean distance between two points.
    :param point: Tuple with the coordinates of the first point.
    :param center: Tuple with the coordinates of the second point.
    :return: The Euclidean distance between the two points.
    >>> _euclid((2,2), (2,2))
    0.0
    >>> _euclid((1, 2), (3, 4))
    2.8284271247461903
    """
    return sqrt(sum((el - center[i]) ** 2 for i, el in enumerate(point)))


def _get_closest_center(point: tuple[float, ...], centers: list[tuple[float, ...]]) -> tuple[float, ...]:
    """
    Get the closest center for a given point. This function calculates the Euclidean distance between the point and each
    center and returns the center with the smallest distance.
    :param point: Tuple with the coordinates of the point.
    :param centers: List of tuples with the coordinates of the centers.
    :return: The center with the smallest Euclidean distance to the point.
    >>> _get_closest_center((1, 2), [(1, 2), (3, 4)])
    (1, 2)
    >>> _get_closest_center((1, 2), [(3, 4), (5, 6)])
    (3, 4)
    """
    distances = [_euclid(point, center) for center in centers]
    min_distance_index = distances.index(min(distances))
    return centers[min_distance_index]


# NUMPY IMPLEMENTATION

def k_means_numpy(data: np.ndarray, initial_centers: np.ndarray) -> set[tuple[float, ...]]:
    """
    Implementation of the k-means algorithm using numpy for improved performance on large datasets. It uses built in
    numpy functions to calculate the distances between the points and the centers. There is no dictionary used to
    store the clusters, instead the clusters are calculated using numpy arrays with extra dimensions.
    :param data: List of tuples with the coordinates of the points.
    :param initial_centers: List of tuples with the coordinates of the initial centroids.
    :return: A set of tuples with the coordinates of the centroids.
    >>> k_means_numpy(np.array([(1, 2), (3, 4), (5, 6), (7, 8)]), np.array([(1, 2), (3, 4)]))
    {(np.float64(2.0), np.float64(3.0)), (np.float64(6.0), np.float64(7.0))}
    """

    # Initialize the centers
    prev_centers: np.ndarray = np.empty(0)
    centers = initial_centers

    # run the k-means algorithm
    while not np.array_equal(prev_centers, centers):
        # Centers to clusters
        distances = np.sqrt(((data - centers[:, np.newaxis]) ** 2).sum(axis=2))
        clusters = np.argmin(distances, axis=0)

        # Clusters to centers
        prev_centers = centers
        centers = np.array([data[clusters == i].mean(axis=0) for i in range(len(centers))])

    return set(tuple(center) for center in centers)


# DASK IMPLEMENTATION

# Check if dask is installed on the system
if importlib.util.find_spec("dask") is not None:

    import dask.array as da


    def k_means_dask(data: da.Array, initial_centers: np.ndarray) -> set[tuple[float, ...]]:
        """
        Implementation of the k-means algorithm using dask for improved performance on large datasets.
        It uses dask arrays to calculate the new centers in parallel. This implementation should be the most efficient
        implementation of the k-means algorithm if the data gets large enough.
        :param data: List of tuples with the coordinates of the points.
        :param initial_centers: List of tuples with the coordinates of the initial centroids.
        :return: A set of tuples with the coordinates of the centroids.
        >>> k_means_dask(da.array([(1, 2), (3, 4), (5, 6), (7, 8)]), np.array([(1, 2), (3, 4)]))
        {(np.float64(2.0), np.float64(3.0)), (np.float64(6.0), np.float64(7.0))}
        """
        # Initialize the centers
        # Use zeros_like to create a dask array with the same shape as the initial centers, this is needed for the
        # comparison in the while loop.
        prev_centers = np.zeros_like(initial_centers)
        centers = initial_centers

        # run the k-means algorithm
        while not da.allclose(prev_centers, centers, atol=1e-5).compute():
            # Centers to clusters
            distances = np.sqrt(((data - centers[:, np.newaxis]) ** 2).sum(axis=2))
            clusters = np.argmin(distances, axis=0)

            # Clusters to centers
            prev_centers = centers
            centers = np.array([data[clusters == i].mean(axis=0) for i in range(len(centers))])

        return set(tuple(center) for center in centers)

if __name__ == "__main__":
    import doctest

    doctest.testmod()

    print(timeit.timeit(lambda: lloyd_algorithm("../data/data_04.txt", 5, True, 2), number=1))
