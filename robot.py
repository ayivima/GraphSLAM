import random

import numpy
from numpy import cos, sin, pi, matrix, zeros
from numpy.linalg import inv


class Robot:

    def __init__(
        self,
        environment,
        sense_range = 30.0,
        motion_noise = 1.0,
        sensor_noise = 1.0,
        timesteps=20
    ):
        self.environment = environment
        self.sense_range = sense_range
        self.x = self.environment.width / 2.0
        self.y = self.environment.height / 2.0
        self.motion_noise = motion_noise
        self.sensor_noise = sensor_noise
        self.timesteps = timesteps
        self.mu = None
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
    
    def _omega_xi_update(
        self,
        tsids1,
        tsids2,
        dvalues,
        noisefactor
    ):
        """Updates Omega and Xi using sensor or motion data.
        
        Arguments
        ---------
        :tsids1: Timestep Indices for X and Y coordinates of robot position
        :tsids2: Timestep Indices X and Y Coordinates of landmark in the case
                  of sensor data, or Next robot destination in the case of 
                  motion data.
                  
        :dvalues: Horizontal and Vertical Distances of robot from landmark
                  in case of sensor data, and for next destination in case 
                  of motion data.
                  
        :noisefactor: The amount of noise to be factored into calculation,
                      to account for robot measurement errors.
                      
                      NB. A given measurement is from a gaussian distribution 
                      which peaks at values close to actual.  
                      
        
        """
        for tsindex1, tsindex2, dvalue in zip(tsids1, tsids2, dvalues):
            self.xi[tsindex1] += dvalue * noisefactor
            self.xi[tsindex2] -= dvalue * noisefactor

            self.omega[tsindex1, tsindex1] -= noisefactor
            self.omega[tsindex1, tsindex2] += noisefactor
            self.omega[tsindex2, tsindex1] += noisefactor
            self.omega[tsindex2, tsindex2] -= noisefactor
    
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

    def navigate(self, stepdist, timesteps=None):
        """Moves robot around to map its assigned environment.

        Arguments
        ---------
        :timesteps: total number of times robot navigates
        :stepdist: the euclidean distance moved for each time step
        """
        
        timesteps = timesteps if timesteps else self.timesteps
        
        def dist_xy():
            # generate a random angle of orientation(theta),
            # and calculate x and y distances for next move
            theta = lambda: random.random() * 2.0 * pi
            return (
                cos(theta()) * stepdist,
                sin(theta()) * stepdist
            )

        navigate_data = []
        num_landmarks = len(self.environment.landmarks)
        all_landmarks_sensed = False

        while not all_landmarks_sensed:

            landmark_states = [0 for landmark_index in range(num_landmarks)]

            # set a random theta
            dx, dy = dist_xy()

            for timestep in range(timesteps-1):
                
                # Set Omega update indexes for destination points
                # and Landmarks
                ptsid_x1 = timestep * 2
                ptsid_y1 = ptsid_x1 + 1
                
                ptsid_x2 = (timestep + 1) * 2
                ptsid_y2 = ptsid_x2 + 1
                
                # collect sensor measurements
                sensed_landmarks = self.sense()
                sense_noise_factor = 1/self.sensor_noise

                # check off all landmarks that were observed
                # And, update Omega and Xi based on sensed landmark data
                for index, xdist, ydist in sensed_landmarks:
                    landmark_states[index] = 1
                    
                    ltsid_x1 = (index + timesteps) * 2
                    ltsid_y1 = ltsid_x1 + 1

                    self._omega_xi_update(
                        (ptsid_x1, ptsid_y1),
                        (ltsid_x1, ltsid_y1),
                        (xdist, ydist),
                        sense_noise_factor
                    )

                # move robot
                while not self.move(dx, dy):
                    # re orient if next destination is outside 
                    # environment area
                    dx, dy = dist_xy()
                
                # update Omega and Xi based on motion data
                motion_noise_factor = 1/self.motion_noise
                self._omega_xi_update(
                    (ptsid_x1, ptsid_y1),
                    (ptsid_x2, ptsid_y2),
                    (dx, dy),
                    motion_noise_factor
                )
                
                # collect/memorize all sensor and motion data
                navigate_data.append([sensed_landmarks, [dx, dy]])

            # we are done when all landmarks were observed; otherwise re-run
            all_landmarks_sensed = sum(landmark_states) == num_landmarks
        
        self.mu = inv(matrix(self.omega)) * self.xi
        
        return navigate_data

    def sense(self):
        """Detects landmarks, records and returns measurements 
        to be used for navigation"""

        # Retrieve noise to be factored in measurements
        noise = lambda: self.getnoise(self.sensor_noise)
        
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
