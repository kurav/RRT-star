from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

np.random.seed(seed= 50)

class Node:
    def __init__(self, row, col):
        self.row = row  # coordinate
        self.col = col  # coordinate
        self.parent = None  # parent node
        self.cost = 0.0  # cost

class RRT:
    # Constructor
    def __init__(self, map_array, start, goal):
        self.map_array = map_array  # map array, 1->free, 0->obstacle
        self.size_row = map_array.shape[0]  # map size
        self.size_col = map_array.shape[1]  # map size

        self.start = Node(start[0], start[1])  # start node
        self.goal = Node(goal[0], goal[1])  # goal node
        self.vertices = []  # list of nodes
        self.found = False
        self.counter = 1

    def init_map(self):
        self.found = False
        self.vertices = []
        self.vertices.append(self.start)

    def dis(self, node1, node2):

        return np.sqrt((node1.row - node2.row) ** 2 + (node1.col - node2.col) ** 2)

    def check_collision(self, node1, node2):

        N = 40
        x_d = (node1.row - node2.row)
        y_d = (node1.col - node2.col)

        x_coordinate = node2.row
        y_coordinate = node2.col
        for i in range(N + 1):

            if (self.map_array[int(x_coordinate)][int(y_coordinate)] == 0):  # 0 is obstacle
                return False
            x_coordinate = node2.row + x_d * (i / N)

            y_coordinate = node2.col + y_d * (i / N)
        return True

    def get_new_point(self, goal_bias):
        goal_bias = 0.025
        rand_prob = np.random.random()
        if rand_prob < goal_bias:
            new_point = Node(self.goal.row, self.goal.col)
        else:
            new_point = Node(np.random.randint(0, self.size_row), np.random.randint(0, self.size_col))
        return new_point

    def get_nearest_node(self, new_point):
        dst = 10000
        for node in self.vertices:
            if (self.dis(node, new_point) < dst):
                dst = self.dis(node, new_point)
                nearest_node = node

        return nearest_node

    def get_neighbors(self, new_node, neighbor_size):
        neighbor_size = 20
        neighbors = []

        for node in self.vertices:
            if node == new_node:
                continue
            if self.dis(new_node, node) < neighbor_size:
                neighbors.append(node)

        return neighbors

    def rewire(self, new_node, neighbors):
        for neighbor_node in neighbors:
            if neighbor_node == new_node.parent:
                continue
            if self.check_collision(neighbor_node, new_node):
                ct_new = new_node.cost + self.dis(new_node, neighbor_node)
                if ct_new < neighbor_node.cost:
                    neighbor_node.parent = new_node
                    neighbor_node.cost = ct_new

    def draw_map(self):
        fig, ax = plt.subplots(1)
        img = 255 * np.dstack((self.map_array, self.map_array, self.map_array))
        ax.set_axis_off()
        ax.imshow(img)

        for node in self.vertices[1:-1]:
            plt.plot(node.col, node.row, markersize=3, marker='o', color='y')
            plt.plot([node.col, node.parent.col], [node.row, node.parent.row], color='y')

        # Draw Final Path if found
        if self.found:
            cur = self.goal
            while cur.col != self.start.col or cur.row != self.start.row:
                plt.plot([cur.col, cur.parent.col], [cur.row, cur.parent.row], color='b')
                cur = cur.parent
                plt.plot(cur.col, cur.row, markersize=3, marker='o', color='b')

        # Draw start and goal
        plt.plot(self.start.col, self.start.row, markersize=5, marker='o', color='g')
        plt.plot(self.goal.col, self.goal.row, markersize=5, marker='o', color='r')

        # show image
        plt.show()

    def draw_map_save_plot(self, save_path="images/plot.png"):

        fig, ax = plt.subplots(1)
        img = 255 * np.dstack((self.map_array, self.map_array, self.map_array))
        ax.set_axis_off()
        ax.imshow(img)

        for node in self.vertices[1:-1]:
            plt.plot(node.col, node.row, markersize=3, marker='o', color='y')
            plt.plot([node.col, node.parent.col], [node.row, node.parent.row], color='y')

        # Draw Final Path if found
        if self.found:
            cur = self.goal
            while cur.col != self.start.col or cur.row != self.start.row:
                plt.plot([cur.col, cur.parent.col], [cur.row, cur.parent.row], color='b')
                cur = cur.parent
                plt.plot(cur.col, cur.row, markersize=3, marker='o', color='b')

        # Draw start and goal
        plt.plot(self.start.col, self.start.row, markersize=5, marker='o', color='g')
        plt.plot(self.goal.col, self.goal.row, markersize=5, marker='o', color='r')

        # Save the image
        save_path = os.path.join("images", f"plot_{self.counter}.png")
        plt.savefig(save_path)
        plt.close(fig)

        self.counter += 1

    def extend_node(self, new_point, nearest_node):
        step_size = 16
        x_new = new_point.row
        y_new = new_point.col

        x_near = nearest_node.row
        y_near = nearest_node.col

        dist = self.dis(new_point, nearest_node)
        v = [x_new - x_near, y_new - y_near]
        new_pos_x = int(x_near + ((step_size * v[0]) / dist))
        new_pos_y = int(y_near + ((step_size * v[1]) / dist))

        if new_pos_x >= self.size_row:
            new_pos_x = self.size_row - 1
        if new_pos_y >= self.size_col:
            new_pos_y = self.size_col - 1
        if new_pos_x < 0:
            new_pos_x = 0
        if new_pos_y < 0:
            new_pos_y = 0

        new_pos = Node(new_pos_x, new_pos_y)

        return new_pos

    def RRT_base(self, n_pts=1000):
        self.init_map()

        for i in range(n_pts):
            # get a new point
            new_point = self.get_new_point(goal_bias=0.05)

            nearest_node = self.get_nearest_node(new_point)
            #if i%3 ==0:
                #self.draw_map_save_plot()
            # self.draw_map()
            if self.dis(new_point, nearest_node) > 12:
                # extend the node and check collision to decide whether to add or drop
                step_node = self.extend_node(new_point, nearest_node)
                new_point = step_node
            if (self.check_collision(new_point, nearest_node)):
                new_point.parent = nearest_node
                new_point.cost = nearest_node.cost + self.dis(new_point, nearest_node)
                self.vertices.append(new_point)
                if new_point == self.goal:
                    self.found = True
            if ((self.dis(new_point, self.goal) <= 2) and self.check_collision(new_point, self.goal)):
                self.found = True
                self.goal.parent = new_point
                self.goal.cost = new_point.cost + self.dis(new_point, self.goal)
                break

        if self.found:
            steps = len(self.vertices) - 2
            length = (self.goal.cost) / 4
            with open('output.txt', 'a') as file:
                file.write(f"{length}\n")

        #self.draw_map()
        #self.draw_map_save_plot()

    def RRT_star(self, n_pts=1000, neighbor_size=20):
        self.init_map()
        for i in range(n_pts):
            new_point = self.get_new_point(goal_bias=0.05)

            # get its nearest node
            nearest_node = self.get_nearest_node(new_point)
            ###################
            #self.draw_map_save_plot()
            if self.dis(new_point, nearest_node) > 16:
                # extend the node and check collision to decide whether to add or drop
                step_node = self.extend_node(new_point, nearest_node)
                new_point = step_node
            if (self.check_collision(new_point, nearest_node)):
                new_point.parent = nearest_node
                new_point.cost = nearest_node.cost + self.dis(new_point, nearest_node)
                self.vertices.append(new_point)
                neighbors = self.get_neighbors(new_point, neighbor_size=20)
                for neighbor_node in neighbors:
                    if neighbor_node == new_point.parent:
                        continue

                    if self.check_collision(neighbor_node, new_point):
                        new_cost = neighbor_node.cost + self.dis(new_point, neighbor_node)

                        if new_cost < new_point.cost:
                            new_point.parent = neighbor_node
                            new_point.cost = new_cost
                self.rewire(new_point, neighbors)

                if new_point == self.goal:
                    self.found = True

            if ((self.dis(new_point, self.goal) <= 2) and self.check_collision(new_point, self.goal)):
                self.found = True
                self.goal.parent = new_point
                self.goal.cost = new_point.cost + self.dis(new_point, self.goal)

        if self.found:
            steps = len(self.vertices) - 2
            length = (self.goal.cost) / 4
            with open('output.txt', 'a') as file:
                file.write(f"{length}\n")


with open('input.txt', 'r') as file:
    start_line = file.readline().strip().split()
    goal_line = file.readline().strip().split()

    start_coords = (float(start_line[0]), float(start_line[1]))
    goal_coords = (float(goal_line[0]), float(goal_line[1]))

    lines = file.readlines()
    coordinates = [line.strip().split() for line in lines]
    coordinates = [(float(x), float(y)) for x, y in coordinates]

with open('output.txt','w') as file:
    pass
x, y = zip(*coordinates)

polygon = patches.Polygon(coordinates, closed=True, edgecolor='black', facecolor='black')

fig, ax = plt.subplots(figsize=(5, 5), dpi=305)  # 30x30 pixels

ax.add_patch(polygon)

# Set the axis limits
ax.set_xlim(-10, 20)
ax.set_ylim(-10, 20)
ax.set_axis_off()

# Save the image with the obstacle in black and the rest in white as a .jpg
plt.savefig('obstacle.jpg', format='jpg', dpi=104, bbox_inches='tight', pad_inches=0, facecolor='white')
plt.close()



def load_map(file_path, resolution_scale):

    img = Image.open(file_path).convert('L')
    # Rescale the image
    size_x, size_y = img.size
    new_x, new_y  = int(size_x*resolution_scale), int(size_y*resolution_scale)
    img = img.resize((new_x, new_y))

    map_array = np.asarray(img, dtype='uint8')
    threshold = 127
    map_array = 1 * (map_array > threshold)

    return map_array


if __name__ == "__main__":
    # Loading coords, then translating and scaling to the image dimensions
    optimal_path  = 28.59
    x_start,y_start = start_coords
    y_start_new = (x_start+10)*4
    x_start_new = 120 - (y_start+10)*4
    x_goal, y_goal = goal_coords
    y_goal_new = (x_goal + 10) * 4
    x_goal_new = 120 - (y_goal + 10) * 4
    start = x_start_new,y_start_new
    goal  = x_goal_new,y_goal_new
    map_array = load_map("obstacle.jpg", 0.3)

    RRT_planner = RRT(map_array, start, goal)
    RRT_planner.RRT_base(n_pts=5000)
    RRT_planner.RRT_star(n_pts=2000)
    with open('output.txt', 'a') as file:
        file.write(f"{optimal_path}")