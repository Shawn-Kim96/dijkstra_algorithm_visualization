"""
input.txt file will be in the same directory file.

1. get the maximum x, y to generate a matrix.
2. make a list of nodes.

file format
- input.txt
total_number_of_nodes
start_node
end_node
start_node, end_node, distance
...

- coord.txt
node_number x y
"""

from collections import defaultdict, deque
import heapq
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('TkAgg')


class DataProcessor:
    def __init__(self):
        self.m = 0
        self.n = 0
        self.start_node = 0
        self.end_node = 0
        self.graph_info = defaultdict(list)  # graph with data structure of hash map; (node, distance)
        self.node_info = [(None, None)]  # node_info[i] = ith node (x, y), 0th index have no meaning
    
    def process_input_files(self):
        # process input.txt
        
        input_info = []
        with open('input.txt', 'r') as f:
            for context in f:
                input_info.append(context.rstrip('\n').split(' '))

        # first, get the information from first 3 lines
        _, self.start_node, self.end_node = input_info[:3]
        self.start_node = int(self.start_node[0])
        self.end_node = int(self.end_node[0])

        # second, get the information about distance from 4th line
        distance_info = input_info[3:]
        distance_info.sort()
        for start, end, distance in distance_info:
            start, end, distance = int(start), int(end), float(distance)
            self.graph_info[start].append((end, distance))
            self.graph_info[end].append((start, distance))

        # process coords.txt
        with open('coords.txt', 'r') as f:
            for context in f:
                x, y = context.rstrip('\n').split(' ')
                x, y = float(x), float(y)
                self.m = max(self.m, x)
                self.n = max(self.n, y)
                self.node_info.append((x, y))
        

    def dijkstras_algorithm(self):
        """
        method: use breadth-first-search to implement dijkstra's algorithm

        1. start with inital point (self.start_node)
        2. iterate the neighborhood node, and append it to queue.
            - Use heapq to always obtain the smallest distance with the path.

        """
        # use heapq (priority queue) to implement this algorithm
        priority_queue = [(0, [self.start_node])]  # (distance: float, path: list)

        # to ignore duplicates, we use graph to check whether we have visited a node.
        distance_history = {node: float('inf') for node in self.graph_info.keys()}
        distance_history[self.start_node] = 0

        result = []

        # add visualize steps.
        visualize_dict = {
            "node_searched": [],  # black
            "node_searching": [],  # blue
        }
        step, previous_distance= 0, 0
        step_nodes = []
        step_edges = []

        while priority_queue:
            current_distance, path = heapq.heappop(priority_queue)
            current_node = path[-1]

            if current_distance > distance_history[current_node]:
                continue
            
            if current_distance > previous_distance:
                step += 1
                previous_distance = current_distance
                self.generate_image_for_step(step, visualize_dict)

                # update step_nodes (move to searched, and initalize searching)
                visualize_dict["node_searched"].extend(visualize_dict["node_searching"])
                visualize_dict["node_searching"] = []
                
                visualize_dict["node_searching"] = [self.node_info[current_node]]
                
            elif current_distance == previous_distance:
                visualize_dict["node_searching"].append(self.node_info[current_node])
                

            distance_history[current_node] = current_distance

            for next_node, diff_distance in self.graph_info[current_node]:
                next_distance = current_distance + diff_distance
                
                if next_node == self.end_node:
                    result.append((next_distance, path + [next_node]))
                
                if next_distance < distance_history[next_node]:
                    distance_history[next_node] = next_distance
                    heapq.heappush(priority_queue, (next_distance, path + [next_node]))
        
        return result
    

    def generate_base_graph_image(self):
        # maximum length will be size of 12
        if self.m > self.n:
            divider = 12 / self.m
            figsize = (12, self.n / divider)
        else:
            divider = 12 / self.n
            figsize = (self.m / divider, 12)
        
        fig, ax = plt.subplots(figsize=figsize)

        for node1, value in self.graph_info.items():
            for node2, _ in value:
                node1_x, node1_y = self.node_info[node1]
                node2_x, node2_y = self.node_info[node2]
                ax.plot([node1_x, node2_x], [node1_y, node2_y], 'b-', linewidth=0.5, alpha=0.2)
        
        for node_num, (x, y) in enumerate(self.node_info[1:]):
            ax.scatter(x, y, s=50, color='tab:blue', alpha=0.5)
            # annotation_distance = 0.3
            # ax.annotate(node_num, (x, y), xytext=(x-annotation_distance, y-annotation_distance))
        
        start_x, start_y = self.node_info[self.start_node]
        end_x, end_y = self.node_info[self.end_node]
        ax.scatter(start_x, start_y, s = 300, color = 'g', alpha=1)
        ax.scatter(end_x, end_y, s = 300, color = 'r', alpha=1)


        return fig, ax
        


    def generate_image_for_step(self, step, visualize_dict):

        # TODO: have to generate a base image (graph with node, road)
        
        fig, ax = self.generate_base_graph_image()
        start_and_end_node = [self.node_info[self.start_node], self.node_info[self.end_node]]
        for state, node_list in visualize_dict.items():
            if state == "node_searched":
                for x, y in node_list:
                    if (x, y) in start_and_end_node:
                        continue
                    ax.scatter(x, y, color = 'tab:gray', s=300, alpha=1)
            elif state == "node_searching":
                for x, y in node_list:
                    if (x, y) in start_and_end_node:
                        continue
                    ax.scatter(x, y, color = 'b', s=300, alpha=1)

        if 'images' not in os.listdir():
            os.mkdir('images')

        fig.savefig(f'images/{step}.png')
        fig.clear()



if __name__ == "__main__":
    dp = DataProcessor()
    dp.process_input_files()
    result = dp.dijkstras_algorithm()
    print(1)