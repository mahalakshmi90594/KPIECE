from abc import ABC, abstractmethod

import numpy as np


class CollisionObject(ABC):
    @abstractmethod
    def isCollision(self, target):
        pass


class CollisionBox(CollisionObject):
    def __init__(self, location, lengths):
        self.location = np.asarray(location)
        self.lengths = np.asarray(lengths)
        self.ndim = self.location.shape[0]

    def isCollision(self, target):
        llimit = self.location
        ulimit = self.location+self.lengths
        print ("Collision Check:", target, llimit, ulimit)
        if (np.all(target>=llimit) and np.all(target<=ulimit)):
            print("Yes Collision")
            return True 
        else:
            print("No Collision")
            return False


class CollisionSphere(CollisionObject):
    def __init__(self, location, radius):
        self.location = np.asarray(location)
        self.radius = radius

    def isCollision(self, target):
        sphere_eqn = sum((target-self.location)**2)
        if sphere_eqn <= self.radius**2:
            return True
        else:
            return False
        
        
