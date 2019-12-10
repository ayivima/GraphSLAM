import random

import numpy
from numpy import cos, pi, sin, zeros
from numpy.linalg import inv


class Robot:

    def __init__(
        self,
        environment,
        sense_range = 30.0,
        motion_noise = 1.0,
        measurement_noise = 1.0,
        timesteps=20
    ):
        self.environment = environment
        self.sense_range = sense_range
        self.x = self.environment.width / 2.0
        self.y = self.environment.height / 2.0
        self.motion_noise = motion_noise
        self.measurement_noise = measurement_noise
        self.timesteps = timesteps
        
        self._omega_xi_init()
    
    def __repr__(self):
        return 'Robot'
    
    def _omega_xi_init(self):
        """Initializes Omega and Xi for calculation of Mu"""
        
        landmark_count = len(self.environment.landmarks)
        rows = cols = 2 * (self.timesteps + landmark_count)

        omega = zeros((rows, cols))
        xi = zeros((rows, 1))

        omega[0, 0] = omega[1, 1] = 1
        xi[0] = self.x
        xi[1] = self.y
        
        self.omega, self.xi = omega, xi
    
    def change_env(self, environment):
        """Places robot in a new environment"""

        self.environment = environment
        self.x = self.environment.width / 2.0
        self.y = self.environment.height / 2.0
    
    def getnoise(self, noise_threshold, limiter=2.0, epsilon=1.0):
        """Calculates and returns random noise based on a set threshold.

        Arguments
        ---------
        :noise_threshold: The noise threshold
        :limiter: Regulates the maximum noise value that can be generated.
        :epsilon: Regulates the range of noise values that can be returned.
        """

        return noise_threshold * (random.random() * limiter - epsilon)

    def move(self, xdist, ydist):
        """Moves robot by specified horizontal and vertical distances.

        Arguments
        ---------
        :xdist: distance to be moved horizontally
        :ydist: distance to be moved vertically
        """

        x = self.x + xdist + self.getnoise(self.motion_noise)
        y = self.y + ydist + self.getnoise(self.motion_noise)
        
        # Check if next destination is within the 
        # robot's assigned environment
        can_move = all([
            x > 0.0,
            y > 0.0,
            x < self.environment.width,
            y < self.environment.height
        ])

        # Move robot if next destination is within assigned environment.
        # Else don't move and return False
        if can_move:
            self.x = x
            self.y = y
            return True
        else:
            return False

    def navigate(self, timesteps, stepdist):
        """Moves robot around to map its assigned environment.

        Arguments
        ---------
        :timesteps: total number of times robot navigates
        :stepdist: the euclidean distance moved for each time step
        """

        def dist_xy():
            # generate a random angle of orientation(theta),
            # and calculate x and y coordinates for next move
            theta = lambda: random.random() * 2.0 * pi
            return (
                cos(theta()) * stepdist,
                sin(theta()) * stepdist
            )

        motion_data = []
        num_landmarks = len(self.environment.landmarks)
        all_landmarks_sensed = False

        while not all_landmarks_sensed:

            landmark_states = [0 for landmark_index in range(num_landmarks)]

            # set a random theta
            dx, dy = dist_xy()

            for timestep in range(self.timesteps-1):

                # collect sensor measurements
                sensed_landmarks = self.sense()

                # check off all landmarks that were observed 
                for index, _, _ in sensed_landmarks:
                    landmark_states[index] = 1

                # move robot
                while not self.move(dx, dy):
                    # re orient if next destination is outside 
                    # environment area
                    dx, dy = dist_xy()

                # collect/memorize all sensor and motion data
                motion_data.append([sensed_landmarks, [dx, dy]])

            # we are done when all landmarks were observed; otherwise re-run
            all_landmarks_sensed = sum(landmark_states) == num_landmarks

        return motion_data

    def sense(self):
        """Detects landmarks, records and returns measurements 
        to be used for navigation"""

        # Retrieve noise to be factored in measurements
        noise = lambda: self.getnoise(self.measurement_noise)
        
        # Function for calculating landmark distances from
        # robot's current position
        dist_xy = lambda x, y: (
            x - self.x + noise(),
            y - self.y + noise() 
        )
        
        # Measure robot distances from sensed landmarks
        measurements = []
        for index, (a, b) in enumerate(self.environment.landmarks):
            xdist, ydist = dist_xy(a, b)

            # Add only measurements for landmarks within sensing range
            if all([
                xdist <= self.sense_range,
                ydist <= self.sense_range
            ]):
                measurements.append([index, xdist, ydist])

        return measurements
