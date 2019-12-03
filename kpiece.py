# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:57:42 2019

@author: mahal
"""
import numpy as np
import scipy.stats as ss
import heapq


class Cell:
    def __init__(self, cell_coord, instantiated_motion, instantiated_iteration):
        self.coord = cell_coord
        self.coverage = 1
        self.instantiated_iteration = instantiated_iteration + 1
        self.score = 1
        self.importance = 0
        self.cntNeighbors = 0
        self.numExpansion = 1

        self.motions = [instantiated_motion]
        self.coverage_motion_cnt = -1
        
    def selectMotion(self):

        num_motions = len(self.motions)
        x = np.arange(num_motions)
        xU, xL = x + 0.5, x - 0.5
        std = np.sqrt(num_motions/3)
        prob = ss.halfnorm.cdf(xU, scale=std) - ss.halfnorm.cdf(xL, scale=std)
        prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
        sorted(prob)
        index = np.random.choice(x, p=prob)

        return self.motions[index]

    def setGoalBias(self, score_init):
        self.score = score_init

    def calcImportance(self):

        self.importance = (np.log(self.instantiated_iteration) * self.score) / (self.numExpansion * (1 + self.cntNeighbors) * self.coverage)

    def calcScore(self, delta_coverage, delta_time):
        P = 0.7 + 5 * (delta_coverage/delta_time)
        self.score = self.score * min(P,1)

    def updateCellMetrics(self, coverage):
        self.coverage += coverage
        self.calcImportance()
            
class Motion:
    def __init__(self, start_position, controls, numSteps, cell, parent_motion, parent_step):
        self.start_position = start_position
        self.controls = controls
        self.numSteps = numSteps
        self.cell = cell

        self.update_parent(parent_motion, parent_step)
        self.children = []

    def add_child(self, child_motion):
        self.children.append(child_motion)

    def update_parent(self, parent_motion, parent_step):
        self.parent = parent_motion
        if self.parent != None:
            self.parent.add_child(self)
        self.parent_step = parent_step
    
    def print_motion(self):
        print("Start: {}, Controls: {}, NumSteps: {}".format(self.start_position, self.controls, self.numSteps))

class Kpiece:                
    def __init__(self, proj_matrix, 
                       grid_dims,
                       obstacles, 
                       start_position,
                       goal_position):
               
        self.proj_matrix = np.array(proj_matrix)
        self.grid_dims = grid_dims
        self.obstacles = obstacles

        self.goal_position = goal_position
        start_motion = Motion(start_position, None, 0, self, None, 0)
        self.motion_tree = start_motion
        self.good_motions = []
        self.good_motion_cells = dict()
        start_cell_coord = self.Coordinate(start_position)
        start_cell = Cell(start_cell_coord, start_motion, 0)

        self.exterior_cells = dict()
        self.exterior_cells[start_cell_coord] = start_cell
        self.interior_cells = dict()

    def Projection(self, q):
        return self.proj_matrix * q
    
    def Coordinate(self, q):
        p = self.Projection(np.array(q))
        return tuple(np.floor(np.divide(p, self.grid_dims)))

    def get_cell_importance(self, cell):
        return cell.importance
    
    def selectCell(self):
        if len(self.interior_cells) == 0:
            return max(list(self.exterior_cells.values()), key=self.get_cell_importance)
        else:
            return max(
                np.random.choice([np.array(self.interior_cells.values()), np.array(self.exterior_cells.values())],
                                 p=[0.25, 0.75]), key=self.get_cell_importance)
    
    def selectNextMotion(self):
        choose_method = np.random.choice([1, 2], p=[0.3, 0.7])
        if choose_method == 2 and len(self.good_motions) != 0:
            popped_motion = heapq.heappop(self.good_motions)
            curr_motion = popped_motion[2]
            curr_cell = curr_motion.cell
            self.good_motion_cells.pop(curr_cell.coord)
        else:
            curr_cell = self.selectCell()
            curr_motion = curr_cell.selectMotion()
        return curr_cell, curr_motion

    def updateNeighboursCnt(self):
        neighbors_direction = ((-1, 0), (0, -1), (1, 0), (0, 1))
        for coord, cell in self.interior_cells.items():
            cell.cntNeighbors = 0
            for n in neighbors_direction:
                neighbor = coord + n
                if neighbor in self.interior_cells or neighbor in self.exterior_cells:
                    cell.cntNeighbors += 1
                    
        for coord, cell in self.exterior_cells.items():
            cell.cntNeighbors = 0
            for n in neighbors_direction:
                neighbor = coord + n
                if neighbor in self.interior_cells or neighbor in self.exterior_cells:
                    cell.cntNeighbors += 1
                if cell.cntNeighbors == 2*2:
                    self.exterior_cells.pop(coord)
                    self.interior_cells[coord] = cell

    def computeGoodMotion(self, motion, end_state):
        motion_cell = motion.cell.coord
        if motion_cell not in self.good_motion_cells:
            curr_dist = np.linalg.norm(np.array(end_state) - np.array(self.goal_position))
            if len(self.good_motions) < 30:
                heapq.heappush(self.good_motions, [curr_dist, id(motion), motion])
                self.good_motion_cells[motion_cell] = 1
            else:
                heap_max = np.argmax(np.array(self.good_motions)[:, 0])
                if self.good_motions[heap_max][0] > curr_dist:
                    replaced_motion = self.good_motions.pop(heap_max)[2]
                    self.good_motion_cells.pop(replaced_motion.cell.coord)
                    heapq.heappush(self.good_motions, [curr_dist, id(motion), motion])
                    self.good_motion_cells[motion_cell] = 1
        
    def splitMotion(self, motion, states, delta_time, iteration_num):
        
        head_motion = motion
        controls = head_motion.controls
        delta_coverage = head_motion.numSteps

        motion.numSteps = 0
        for i in range(1, len(states)):
            curr_cell_coord = self.Coordinate(states[i])
            if motion.cell.coord == curr_cell_coord:
                motion.numSteps += 1
            else:
                motion.numSteps += 1
                self.computeGoodMotion(motion, states[i-1])
                if(len(motion.cell.motions) == 1):
                    motion.cell.setGoalBias(np.linalg.norm(list(np.subtract(states[i-1], self.goal_position))))
                motion.cell.updateCellMetrics(motion.numSteps)
                
                new_motion = Motion(states[i], controls, 0, None, motion, motion.numSteps)
                if curr_cell_coord in self.interior_cells:
                    curr_cell = self.interior_cells[curr_cell_coord]
                    curr_cell.motions.append(new_motion)
                    curr_cell.numExpansion += 1
                elif curr_cell_coord in self.exterior_cells:
                    curr_cell = self.exterior_cells[curr_cell_coord]
                    curr_cell.motions.append(new_motion)
                    curr_cell.numExpansion += 1
                else:
                    curr_cell = Cell(curr_cell_coord, new_motion, iteration_num)
                    self.exterior_cells[curr_cell_coord] = curr_cell

                new_motion.cell = curr_cell
                motion = new_motion
             
        motion.cell.updateCellMetrics(motion.numSteps)
        self.updateNeighboursCnt()

        head_motion.cell.calcScore(delta_coverage, delta_time)

        return motion

