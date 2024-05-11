"""
This project uses two algorithms, Nearest Neighbors, and Ant System, to solve
the Traveling Salesman Problem. This program will generate 50 cities at random locations and
will compare which one will result in the more optimal path. It will output the distance as
well as an image to see which is visually more optimal.

run these commands in order to run the program:
1. python3 -m venv .venv
2. cd ./.venv/bin/
3. source activate
4. pip install networkx[default]
6. python3 main.py

"""

import random
import math
import networkx as nx
import matplotlib.pyplot as plt
import copy

# Number of cities
CITY = 50

# Seed for reproducibility
random.seed(0)


# Class to represent a graph and related operations
class Graph:
    def __init__(self, point):
        """
        Initializes the graph with given points.

        Parameters:
        - point: Dictionary containing points with their coordinates.
        """
        self.G = nx.Graph()
        self.point = point
        for vertex in point:
            self.G.add_node(vertex, pos=point[vertex])

    def getTotalDistance(self):
        """
        Calculates the total distance of edges in the graph.

        Returns:
        - Total distance of edges in the graph.
        """
        sum = 0
        # Get weights of edges and calculate their sum
        weights = nx.get_edge_attributes(self.G, "weight").values()

        for weight in weights:
            sum += weight
        return sum

    def visualizeGraph(self, name):
        """
        Visualizes the graph and saves the plot to a file.

        Parameters:
        - name: Name of the file to save the plot.
        """
        pos = nx.get_node_attributes(self.G, "pos")
        plt.figure(figsize=(10, 10))
        nx.draw_networkx(self.G, pos)
        plt.savefig(name)

    def setNearestNeighbors(self):
        """
        Sets edges between nodes based on nearest neighbor algorithm.
        """
        point = copy.deepcopy(self.point)
        currIdx = 0
        initialPoint = currPoint = point[currIdx]
        del point[0]  # Remove initial point from the list
        while True:
            shortestDistance = float("inf")
            shortestIdx = -1
            shortestPoint = (float("inf"), float("inf"))
            if len(point) == 0:
                shortestDistance = getDistance(currPoint, initialPoint)
                self.G.add_edge(currIdx, 0, weight=shortestDistance)
                break
            # Find nearest neighbor
            for tempIdx in point:
                tempPoint = point[tempIdx]
                tempDistance = getDistance(currPoint, tempPoint)
                if tempDistance < shortestDistance:
                    shortestDistance = tempDistance
                    shortestIdx = tempIdx
                    shortestPoint = tempPoint
            # Add edge between current node and nearest neighbor
            self.G.add_edge(currIdx, shortestIdx, weight=shortestDistance)
            currIdx = shortestIdx
            currPoint = shortestPoint
            del point[currIdx]  # Remove the nearest neighbor from the list

    def setACO(self, ants=50):
        """
        Applies Ant Colony Optimization (ACO) algorithm to find the shortest path.

        Parameters:
        - ants: Number of ants to be used in the algorithm.

        Returns:
        - Total distance of the best path found.
        - Best path found by the algorithm.
        """
        points = copy.deepcopy(self.point)
        pheremoneGraph = copy.deepcopy(self.G)
        iteration = 500
        evaporationRate = 0.2
        alpha = 1
        beta = 2

        nodes = dict(pheremoneGraph.nodes(data=True))

        # Function to construct solutions for each ant
        def constructAntSolutions(startingAntPos):
            temp_points = dict(pheremoneGraph.nodes(data="pos"))
            visitedNodes = set([startingAntPos])
            prevVisitedNodes = set()
            currAntPos = startingAntPos
            totalWeight = 0
            path = []

            # Helper function to filter visited nodes
            def filter_node(visitedNodes):
                return lambda n: n not in visitedNodes

            # Construct path for the ant
            while len(visitedNodes) != len(temp_points):
                distances = []
                # Calculate distances to unvisited nodes
                for point in temp_points:
                    if point in visitedNodes:
                        continue
                    distance = getDistance(temp_points[currAntPos], temp_points[point])
                    distances.append((distance, point))

                # Consider only unvisited edges for the current ant
                view = nx.subgraph_view(
                    pheremoneGraph, filter_node=filter_node(prevVisitedNodes)
                )
                # Calculate probabilities based on pheromones and distances
                pheremones = list(view.edges(currAntPos, data="weight"))
                temp = list(zip(distances, pheremones))
                totalDistance = 0
                for dist, edge in temp:
                    totalDistance += (edge[2] ** alpha) * ((1 / dist[0]) ** beta)

                # Calculate probabilities
                weights = []
                for dist, edge in temp:
                    weight = (
                        (edge[2] ** alpha) * ((1 / dist[0]) ** beta) / totalDistance
                    )
                    weights.append(weight)
                (distance, p) = random.choices(population=distances, weights=weights)[0]

                # Update path and visited nodes
                path.append((currAntPos, p))
                prevVisitedNodes.add(currAntPos)
                visitedNodes.add(p)
                currAntPos = p
                totalWeight += distance

            # Connect last two nodes to complete the path
            path.append((currAntPos, startingAntPos))
            distance = getDistance(temp_points[currAntPos], temp_points[startingAntPos])
            totalWeight += distance

            return totalWeight, path

        # Function to update pheromone levels
        def updatePheromones(totalDistance, path):
            inverseTotalWeight = 1 / totalDistance
            for u, v in path:
                pheremone = pheremoneGraph.edges[u, v]["weight"]
                newWeight = pheremone + inverseTotalWeight
                pheremoneGraph.update(edges=[(u, v, {"weight": newWeight})])

            # Evaporate pheromones
            for u, v, pheremone in pheremoneGraph.edges(data="weight"):
                pheremoneGraph.update(
                    edges=[(u, v, {"weight": (1 - evaporationRate) * pheremone})]
                )

            return

        # initialize pheremone matrix with a weight of 1.0
        # pheremone matrix is global for all ants
        for u in nodes:
            for v in nodes:
                if u == v:
                    continue
                pheremoneGraph.add_edge(u, v, weight=1.0)
        bestDistance = float("inf")
        bestPath = []
        # Run ACO algorithm for a certain number of iterations
        for _ in range(iteration):
            for a in range(ants):
                val = random.randint(0, len(points) - 1)
                totalDistance, path = constructAntSolutions(val)
                # Update best path if a better one is found
                if totalDistance < bestDistance:
                    bestDistance = totalDistance
                    bestPath = path
                updatePheromones(totalDistance, path)
        # Add best path to the graph
        self.G.add_edges_from(bestPath)
        return bestDistance, bestPath


# generate a random x y coordinate
def randomPoint(min=0, max=1000):
    x = random.randint(min, max)
    y = random.randint(min, max)
    return (x, y)


# calculate the euclidean distance of two points from x,y tuple
def getDistance(p1, p2):
    return math.sqrt(math.pow(p2[0] - p1[0], 2) + math.pow(p2[1] - p1[1], 2))


def main():
    # generate 50 random points on graph
    point = {}
    for v in range(50):
        point[v] = randomPoint()
    NNgraph = Graph(point)

    # call nearest neighbors function to calculate its solution from the 50 points
    NNgraph.setNearestNeighbors()
    totalDistance = NNgraph.getTotalDistance()
    NNgraph.visualizeGraph("nearest-neighbors-graph.png")
    print("total distance: ", totalDistance)

    # call ACO function to calculate its solution from the 50 points
    ACOgraph = Graph(point)
    totalDistance, a = ACOgraph.setACO()
    ACOgraph.visualizeGraph("aco-graph.png")
    print("total distance: ", totalDistance)

    return 0


main()
