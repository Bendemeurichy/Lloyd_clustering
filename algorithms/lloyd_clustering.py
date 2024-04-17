# https://dodona.be/nl/courses/3363/series/36091/activities/207273748/#
# Date: 14/04/24
# Description: Lloyd's clustering algorithm
import random
from math import sqrt
import ast
import importlib.util
import numpy as np


def lloyd_algorithm(data: str, k: int, k_plus_plus_init: bool = False, variant: int = 2) -> set[tuple[float, ...]]:
    """
    Implementation of the lloyd clustering algorithm using the k-means algorithm.
    We're using 3 versions of the k-means algorithm:
    :param variant: 1 -> base implementation, 2 -> numpy implementation
    :param data: Path to the file with the data.
    :param k: Amount of clusters wanted.
    :param k_plus_plus_init: Boolean to indicate if the k-means++ initialization should be used or the first k points.
    :return: A set of tuples with the coordinates of the centroids.
    """

    # Read the data from the file
    with open(data, encoding="utf-8") as file:
        filecontent = file.read()
        points: list[tuple[float, ...]] = list(ast.literal_eval(filecontent))

    # Initialize the centroids
    if k_plus_plus_init:
        initial_centers = _k_means_plus_plus(points, k)
    else:
        initial_centers = points[:k]

    # Run the desired variant of the k-means algorithm
    if variant == 1:
        return _k_means_base(points, initial_centers)

    return _k_means_numpy(np.array(points), np.array(initial_centers))


def _k_means_plus_plus(data: list[tuple[float, ...]], k: int) -> list[tuple[float, ...]]:
    """
    Implementation of the k-means++ initialization.
    :param data: List of tuples with the coordinates of the points.
    :param k: Amount of clusters wanted.
    :return: A set of tuples with the coordinates of the centroids.
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


def _k_means_base(data: list[tuple[float, ...]], initial_centers: list[tuple[float, ...]]) -> set[tuple[float, ...]]:
    """
    Implementation of the k-means algorithm.
    :param data: List of tuples with the coordinates of the points.
    :param initial_centers: List of tuples with the coordinates of the initial centroids.
    :return: A set of tuples with the coordinates of the centroids.
    """

    prev_centers: list[tuple[float, ...]] = []
    centers = initial_centers
    while prev_centers != centers:
        # Centers to clusters

        clusters: dict[tuple[float, ...], list[tuple[float, ...]]] = {center: [] for center in centers}
        for point in data:
            closest_center = get_closest_center(point, centers)
            clusters[closest_center].append(point)

        # Clusters to centers
        new_centers = []
        for center, points in clusters.items():
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
    """
    return sqrt(sum((el - center[i]) ** 2 for i, el in enumerate(point)))


def get_closest_center(point, centers):
    distances = [_euclid(point, center) for center in centers]
    min_distance_index = distances.index(min(distances))
    return centers[min_distance_index]


def _k_means_numpy(data: np.ndarray, initial_centers: np.ndarray) -> set[tuple[float, ...]]:
    """
    Implementation of the k-means algorithm using numpy for improved performance on large datasets.
    :param data: List of tuples with the coordinates of the points.
    :param initial_centers: List of tuples with the coordinates of the initial centroids.
    :return: A set of tuples with the coordinates of the centroids.
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


def _k_means_dask(data: np.ndarray, initial_centers: np.ndarray) -> set[tuple[float, ...]]:
    """
    Implementation of the k-means algorithm using dask for improved performance on large datasets.
    :param data: List of tuples with the coordinates of the points.
    :param initial_centers: List of tuples with the coordinates of the initial centroids.
    :return: A set of tuples with the coordinates of the centroids.
    """
    pass
