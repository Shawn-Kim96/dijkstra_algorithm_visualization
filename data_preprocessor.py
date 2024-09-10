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


class DataProcessor:
    def __init__(self):
        self.m = 0
        self.n = 0
        self.start_node = 0
        self.end_node = 0
        self.graph_info = defaultdict(list)  # graph with data structure of hash map
        self.node_info = [(None, None)]  # node_info[i] = ith node (x, y), 0th index have no meaning
    
    def process_input_files(self):
        # process input.txt
        input_info = []
        with open('input.txt', 'r') as f:
            for context in f:
                input_info.append(context.rstrip('\n').split(' '))

        _, self.start_node, self.end_node = input_info[:3]
        for start, end, distance in input_info[3:]:
            start, end, distance = int(start), int(end), float(distance)
            self.graph_info[start].append((distance, end))
            self.graph_info[end].append((distance, start))

        # process coords.txt
        with open('coords.txt', 'r') as f:
            for context in f:
                # context = f.read()
                
                x, y = context.rstrip('\n').split(' ')
                x, y = float(x), float(y)
                self.m = max(self.m, x)
                self.n = max(self.n, y)
                self.node_info.append((x, y))

    # def dijkstras_algorithm(self):
    #     # use breadth-first-search for this problem

    #     queue = deque([])
        

if __name__ == "__main__":
    dp = DataProcessor()
    dp.process_input_files()
    print(dp.node_info)
    print(dp.graph_info)