"""
Implements an Environment class which generates 2D environments with landmarks (or obstacles).
"""


import math
import random


__author__ = "Victor Mawusi Ayi <ayivima@hotmail.com>"


class Environment():

    def __init__(self, size, landmark_count):
        """Intializes environment state"""

        # Set environment dimensions
        if isinstance(size, (int, float)):
            self.width = self.height = size

        elif isinstance(size, (tuple, list, set)):
            self.width, self.height = size
            
            if not all((
                isinstance(self.width, (int, float)),
                isinstance(self.height, (int, float))
            )):
                raise ValueError(
                    "Size must be a sequence with two "
                    "numbers of type integer or float."
                )
        
        if landmark_count >= (self.width * self.height) * 2/3:
            raise ValueError(
                "landmark_count argument must be less than "
                "two-thirds of the area of the environment"
            )
             
        self.landmark_count = landmark_count
        
        # Set environment landmarks
        self._set_landmarks_()
        
    def _set_landmarks_(self, seed=None):
        """Sets new landmarks"""
        
        random.seed(seed)
        
        randx = lambda: int(random.random() * self.width)
        randy = lambda: int(random.random() * self.height)
        
        # Initialize landmarks
        self.landmarks = [
            (randx(), randy()) for _ in range(self.landmark_count)
        ]
