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

from collections import defaultdict
import heapq
import matplotlib.pyplot as plt
import os
import cv2
import argparse

SJSU_ID = "018219422"


class DataProcessor:
    def __init__(self, make_video: bool, steps_per_frame: int):
        self.m = 0
        self.n = 0
        self.start_node = 0
        self.end_node = 0
        self.graph_info = defaultdict(list)  # graph with data structure of hash map; (node, distance)
        self.node_info = [(None, None)]  # node_info[i] = ith node (x, y), 0th index have no meaning
        self.fig, self.ax = None, None  # base background image for video
        self.node_scatters = {}
        self.make_video = make_video
        self.steps_per_frame = int(steps_per_frame)
    
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
        distance_info.sort()  # to maintain node order (starting from small)
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
        priority_queue = [(0, [self.start_node], [0])]  # (distance: float, path: list, distance_history: list)

        # to ignore duplicates, we use graph to check whether we have visited a node.
        min_distance_from_start = defaultdict(lambda: float('inf'))
        min_distance_from_start[self.start_node] = 0.0

        result = []  # final optimized distance path

        # add visualize steps.
        visualize_dict = {
            "node_searched": [],  # black
            "node_searching": [],  # blue
        }
        step, previous_distance= 0, 0

        while priority_queue:
            current_distance, path, distance_history = heapq.heappop(priority_queue)
            current_node = path[-1]

            min_distance_from_start[current_node] = current_distance

            # update visualize_dict for images
            if current_distance > previous_distance:
                step += 1
                previous_distance = current_distance

            # generate video for every steps_per_frame
            visualize_dict["node_searching"].append(self.node_info[current_node])
            if step % self.steps_per_frame == 0:
                self.generate_image_for_step(step, visualize_dict)
                # update step_nodes (move to searched, and initalize searching)
                visualize_dict["node_searched"].extend(visualize_dict["node_searching"])
                visualize_dict["node_searching"] = []
                
                # visualize_dict["node_searching"] = [self.node_info[current_node]]
                step += 1
                
            
            for next_node, diff_distance in self.graph_info[current_node]:
                next_distance = current_distance + diff_distance
                distance_history = distance_history + [next_distance]
                
                if next_node == self.end_node and not result:
                    result = (path + [next_node], distance_history)
                
                if next_distance < min_distance_from_start[next_node]:
                    min_distance_from_start[next_node] = next_distance
                    heapq.heappush(priority_queue, (next_distance, path + [next_node], distance_history))
        
        return result
    

    def generate_base_graph_image(self):
        # maximum length will be size of 12
        if self.fig is not None and self.ax is not None:
            return

        if self.m > self.n:
            divider = 12 / self.m
            figsize = (12, self.n * divider)
        else:
            divider = 12 / self.n
            figsize = (self.m * divider, 12)
        
        self.fig, self.ax = plt.subplots(figsize=figsize)

        for node1, neighbors_info in self.graph_info.items():
            node1_x, node1_y = self.node_info[node1]
            for node2, _ in neighbors_info:
                node2_x, node2_y = self.node_info[node2]
                self.ax.plot([node1_x, node2_x], [node1_y, node2_y], 'b-', linewidth=0.5, alpha=0.2)
        
        for node_num, (x, y) in enumerate(self.node_info[1:]):
            self.ax.scatter(x, y, s=50, color='tab:blue', alpha=0.5)
            # annotation_distance = 0.3
            # ax.annotate(node_num, (x, y), xytext=(x-annotation_distance, y-annotation_distance))
        
        start_x, start_y = self.node_info[self.start_node]
        end_x, end_y = self.node_info[self.end_node]
        self.ax.scatter(start_x, start_y, s = 300, color = 'g', alpha=1)
        self.ax.scatter(end_x, end_y, s = 300, color = 'r', alpha=1)
        

    def generate_image_for_step(self, step, visualize_dict):        
        self.generate_base_graph_image()
        start_and_end_node = [self.node_info[self.start_node], self.node_info[self.end_node]]

        for state, node_list in visualize_dict.items():
            if state == "node_searched":
                for x, y in node_list:
                    if (x, y) in start_and_end_node:
                        continue
                    self.ax.scatter(x, y, color = 'tab:gray', s=300, alpha=1)
            elif state == "node_searching":
                for x, y in node_list:
                    if (x, y) in start_and_end_node:
                        continue
                    self.ax.scatter(x, y, color = 'b', s=300, alpha=1)

        if 'images' not in os.listdir():
            os.mkdir('images')

        self.fig.savefig(f'images/{step:04d}.png')
        # self.fig.clear()
    

    def generate_final_image(self, optimized_path):
        self.fig, self.ax = None, None
        self.generate_base_graph_image()
        for i in range(len(optimized_path)-1):
            path_start_node, path_end_node = optimized_path[i], optimized_path[i+1]
            node1_x, node1_y = self.node_info[path_start_node]
            node2_x, node2_y = self.node_info[path_end_node]
            self.ax.plot([node1_x, node2_x], [node1_y, node2_y], 'r-', linewidth=3, alpha=1)

        self.fig.savefig(f'images/final.png')
        self.fig.clear()

    
    def generate_output_file(self, path_history, distance_history):
        node_info_string = " ".join([str(x) for x in path_history])
        distance_info_string = " ".join([f"{x:.5f}" for x in distance_history])

        with open(f"{SJSU_ID}.txt", "w") as file:
            file.write(node_info_string + '\n' + distance_info_string)


    def generate_video_from_images(self):
        image_path_list = sorted([f"images/{x}" for x in os.listdir('images/') if '.png' in x])
        image_list = [cv2.imread(x) for x in image_path_list]
        
        height, width, layers = image_list[0].shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 video
        video = cv2.VideoWriter(f"{SJSU_ID}.mp4", fourcc, 10, (width, height))

        # Iterate through each image and write it to the video
        for image in image_list:
            video.write(image)
        
        # Release the video writer
        video.release()
        
        
    def main(self):
        self.process_input_files()
        optimized_path, optimized_distance_history = self.dijkstras_algorithm()
        self.generate_final_image(optimized_path)
        self.generate_output_file(optimized_path, optimized_distance_history)
        self.generate_video_from_images()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dijktar Algorithm (#018219422)")
    parser.add_argument('--video', type=int, default=1, help="Set to 1 to generate video, 0 otherwise.")
    parser.add_argument('--steps_per_frame', type=int, default=3, help="Increase value to include more steps in one frame. Default = 3")
    args = parser.parse_args()

    dp = DataProcessor(make_video=bool(args.video), steps_per_frame=args.steps_per_frame)
    # dp = DataProcessor(make_video=True, steps_per_frame=3)
    dp.main()
