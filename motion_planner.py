import datetime
import random
import numpy as np
from simulation import Simulation
from collision import CollisionBox, CollisionSphere
from kpiece import Kpiece, Motion

class MotionPlanning:
    def __init__(self, dim_ranges,
                 grid_dims,
                 start_position,
                 goal_position,
                 control_ranges,
                 proj_matrix,
                 obstacles,
                 n_iteration):
        self.dim_ranges = dim_ranges
        self.control_ranges = control_ranges
        self.simulator = Simulation(dim_ranges, grid_dims, obstacles, start_position, goal_position)
        self.kpiece = Kpiece(proj_matrix, grid_dims, obstacles, start_position, goal_position)

        self.n_iteration = n_iteration

    def run(self):
        for i in range(self.n_iteration):
            curr_cell, curr_motion = self.kpiece.selectNextMotion()
            curr_state, curr_motion_step = self.selectState(curr_motion)
            curr_control = self.selectControl()
            numSteps = random.sample(range(1, 15), 1)[0]

            curr_cell.numExpansion += 1
            new_motion = Motion(curr_state, curr_control, numSteps, curr_cell, None, 0)
            start = datetime.datetime.now()
            states, isEnd = self.simulator.simulateMotion(new_motion)
            delta_time = (datetime.datetime.now() - start).microseconds
            if (new_motion.numSteps != 0 or delta_time != 0):
                new_motion.update_parent(curr_motion, curr_motion_step)
                last_motion = self.kpiece.splitMotion(new_motion, states, delta_time, i)

            if isEnd:
                print("End of Exploration...Bactracing...")
                return self.backTrace(last_motion)
        return None

    def selectState(self, motion):
        states = self.simulator.generateStates(motion)
        motion_step = np.random.choice(len(states))
        return states[motion_step], motion_step

    def selectControl(self):
        direction_x = np.random.choice(np.arange(self.control_ranges[0][0], self.control_ranges[0][1]))
        direction_y = np.random.choice(np.arange(self.control_ranges[1][0], self.control_ranges[1][1]))
        return [direction_x, direction_y]

    def backTrace(self, last_motion):
        motions_list = []

        curr_motion = last_motion
        prev_motion = None
        while curr_motion != None:
            if prev_motion != None:
                self.simulator.drawMotion(prev_motion.start_position, curr_motion.start_position, isForward=False)
            motions_list.append(curr_motion)
            prev_motion = curr_motion
            curr_motion = curr_motion.parent
        return motions_list


if __name__ == '__main__':
    dim_ranges = [500, 500]
    grid_dims = [20, 20]
    start_position = [400, 50]
    goal_position = [50, 400]
    control_ranges = [[-5, 5], [-5, 5]]  # [Velocity_x per time step, Velocity_y per time step]
    proj_matrix = [1, 1]
    obstacles = [
        CollisionBox((100, 100), (50, 50)),
        CollisionBox((200, 200), (50, 50)),
        CollisionSphere((200,200), 50)
    ]
    n_iteration = 1000

    motion_planner = MotionPlanning(dim_ranges,
                                    grid_dims,
                                    start_position,
                                    goal_position,
                                    control_ranges,
                                    proj_matrix,
                                    obstacles,
                                    n_iteration)
    motions = motion_planner.run()
    for i in motions:
        i.print_motion()
        if i.parent != None:
            i.parent.numSteps = i.parent_step

    input("Enter any key to quit.")