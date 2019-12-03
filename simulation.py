import pygame
from collision import CollisionBox, CollisionSphere
import numpy as np

class Simulation:
    def __init__(self, dim_range, grid_dims, obstacles, start_position, goal_position, goal_precision=5.0):
        self.width = dim_range[0]
        self.height = dim_range[1]
        self.grid_dims = grid_dims
        self.propagation_stepsize = 1
        self.obstacles = obstacles
        self.goal_position = goal_position
        self.goal_precision = goal_precision

        pygame.init()
        pygame.display.set_caption("Car tutorial")
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.screen.fill((255, 255, 255))
        for i in range(self.height // 20):
            pygame.draw.line(self.screen, [0] * 3, (0, self.grid_dims[0] * i), (640, self.grid_dims[1] * i))

        for i in range(self.width // 20):
            pygame.draw.line(self.screen, [0] * 3, (self.grid_dims[0] * i, 0), (self.grid_dims[1] * i, 480))

        pygame.draw.circle(self.screen, [0, 255, 0], start_position, 5)
        pygame.draw.circle(self.screen, [255, 0, 0], goal_position, 5)

        for obs in self.obstacles:
            if isinstance(obs, CollisionBox):
                pygame.draw.rect(self.screen, [0, 0, 0],
                                 pygame.Rect(obs.location[0], obs.location[1], obs.lengths[0], obs.lengths[1]))
            elif isinstance(obs, CollisionSphere):
                pygame.draw.circle(self.screen, [0, 0, 0], obs.location, obs.radius)

        self.clock = pygame.time.Clock()
        self.ticks = 60

    def get_PropagationStepsize(self):
        return self.propagation_stepsize

    def generateStates(self, motion):
        if motion.controls == None:
            return [motion.start_position]
        return self.simulateMotion(motion, isdrawMotions=False)[0]

    def drawMotion(self, prev_state, curr_state, isForward=True):
        color = [0, 0, 255] if isForward else [255, 0, 0]
        pygame.draw.line(self.screen, color, prev_state,
                         curr_state)  # draw motion line from prev position to curr position
        pygame.display.flip()

    def simulateMotion(self, motion, isdrawMotions=True):
        position = motion.start_position
        direction = motion.controls
        States = [position]
        isEnd = False

        for i in range(motion.numSteps):
            position = position + (self.propagation_stepsize * np.array(direction))

            if position[0] < 0 or position[1] < 0 or position[0] >= self.width or position[1] >= self.height:
                motion.numSteps = i
                break

            isCollisionBreak = False
            if self.obstacles != None:
                for obstacle in self.obstacles:
                    if obstacle.isCollision(position):
                        motion.numSteps = i
                        isCollisionBreak = True
                        break
            if isCollisionBreak:
                break
            States.append(position)

            if isdrawMotions:
                self.drawMotion(States[-2], States[-1])  # draw motion line from prev position to curr position

            if np.linalg.norm(self.goal_position - position) <= self.goal_precision:
                motion.numSteps = i
                isEnd = True
                break
            self.clock.tick(self.ticks)
        return States, isEnd