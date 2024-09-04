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
import logging

m, n = 0, 0
node_info = []
with open('coords.txt', 'r') as f:
    context = f.read()
    node_num, x, y = context.split(' ')
    m = max(m, x)
    n = max(n, y)
    node_info.append((m, n))

input_info = []

with open('input.txt', 'r') as f:
    context = f.read()
    input_info.append(context.split(' '))

total_number_of_nodes, start_node, end_node = input_info[:3]
graph = [(start, end, distance) for start, end, distance in input_info[3:]]



class DataProcessor:
    def __init__(self):
        self.m = 0
        self.n = 0
        self.start_node = 0
        self.end_node = 0
        self.graph_info = []
        self.node_info = []
    
    def process_input_files(self):
        # process input.txt
        with open('input.txt', 'r') as f:
            context = f.read()
            input_info.append(context.split(' '))

        _, self.start_node, self.end_node = input_info[:3]
        self.graph_info = [(start, end, distance) for start, end, distance in input_info[3:]]

        # process coords.txt
        with open('coords.txt', 'r') as f:
            context = f.read()
            node_num, x, y = context.split(' ')
            self.m = max(self.m, x)
            self.n = max(self.n, y)
            self.node_info.append((node_num, x, y))
