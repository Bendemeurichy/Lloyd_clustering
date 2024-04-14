# https://dodona.be/nl/courses/3363/series/36091/activities/207273748/#
# Date: 14/04/24
# Description: Lloyd's clustering algorithm
import random
from math import sqrt
import ast

# TODO: make possible for n-dimensional data instead of 2D data


def lloyd_algorithm(data: str, k: int, k_plus_plus_init: bool = False) -> set[tuple[float, float]]:
    """
    Implementation of the lloyd clustering algorithm using the k-means algorithm.
    We're using 3 versions of the k-means algorithm:
    :param data: Path to the file with the data.
    :param k: Amount of clusters wanted.
    :param k_plus_plus_init: Boolean to indicate if the k-means++ initialization should be used or the first k points.
    :return: A set of tuples with the coordinates of the centroids.
    """

    # Read the data from the file
    with open(data, 'utf8') as file:
        filecontent = file.read()
        data = ast.literal_eval(filecontent)

    # Initialize the centroids
    if k_plus_plus_init:
        initial_centers = _k_means_plus_plus(data, k)
    else:
        initial_centers = data[:k]

    # Run the k-means algorithm
    return _k_means_base(data, initial_centers)


def _k_means_plus_plus(data: list[tuple[float, float]], k: int) -> set[tuple[float, float]]:
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
        distances = [min([_euclid(point, center) for center in centers]) for point in data]

        # Calculate the probability of each point to be the next center
        probabilities = [distance ** 2 for distance in distances]
        sum_probabilities = sum(probabilities)
        probabilities = [probability / sum_probabilities for probability in probabilities]

        # Select the next center randomly
        centers.append(random.choices(data, probabilities)[0])

    return set(centers)


def _k_means_base(data: list[tuple[float, float]], initial_centers: [tuple[float, float]]) -> set[tuple[float, float]]:
    """
    Implementation of the k-means algorithm.
    :param data: List of tuples with the coordinates of the points.
    :return: A set of tuples with the coordinates of the centroids.
    """

    prev_centers = []
    centers = initial_centers
    while prev_centers != centers:
        # Centers to clusters
        clusters = {center: [] for center in centers}
        for point in data:
            closest_center = min(centers, key=lambda center: _euclid(point, center))
            clusters[closest_center].append(point)

        # Clusters to centers
        new_centers = []
        for center, points in clusters.items():
            new_center = (
                sum([point[0] for point in points]) / len(points), sum([point[1] for point in points]) / len(points))
            new_centers.append(new_center)

        prev_centers = centers
        centers = new_centers

    return set(centers)


def _distortion(data: list[tuple[float, float]], centers: list[tuple[float, float]]) -> float:
    """
    Calculate the distortion of the clustering.
    This is equal to the average of the minimal distance between each point and the centroids.
    :param data: List of tuples with the coordinates of the points.
    :param centers: Set of tuples with the coordinates of the centroids.
    :return: The distortion of the clustering.
    """

    return (1 / len(data)) * sum([min([_euclid(point, center) for center in centers]) for point in data])


def _euclid(point: tuple[float, float], center: tuple[float, float]) -> float:
    """
    Calculate the euclidean distance between two points.
    :param point: Tuple with the coordinates of the first point.
    :param center: Tuple with the coordinates of the second point.
    :return: The euclidean distance between the two points.
    """

    return sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)
