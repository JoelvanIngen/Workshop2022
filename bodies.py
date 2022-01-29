import math
import numpy as np

import logging
# set-up logger
logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)

c_handler = logging.StreamHandler()
c_handler.setLevel(logging.DEBUG)
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)

logger.debug('Logger initialized')


class Star():
    def __init__(self, radius, mass):
        self.radius = radius
        self.radius_sqr2 = math.pow(radius, 2)
        self.mass = mass

    def initialize(self, planet, labdas):
        self.x = planet.a * planet.e
        self.y = 0  # for now
        self.labdas = labdas
        self.tot_intensities = []

        for labda in self.labdas:
            if not labda:
                self.tot_intensities.append(math.pi * self.radius_sqr2)
            else:
                self.tot_intensities.append(self.calcLimbDarkening(labda))

    def calcLimbDarkening(self, labda):
        inverse_labda = 1000/labda
        dr = 0.001
        
        min_xi_2 = -0.231961
        min_xi_3 = -0.0772776
        min_xi_4 = -0.0429718

        if 303 <= labda <= 367:
            c_0 = 0.3998 - 0.10256 * inverse_labda
            c_2 = -0.0063 + 0.0006839 * math.pow(inverse_labda,5)
            c_3 = -0.2291 - 0.0020539 * math.pow(inverse_labda,5)
            c_4 = 0.3324 + 0.001083 * math.pow(inverse_labda,5)
        elif 372 <= labda <= 405:
            c_0 = 0.2102 - 0.03570 * inverse_labda
            c_2 = -0.3373 + 0.0043378 * math.pow(inverse_labda,5)
            c_3 = 1.6731 - 0.0206681 * math.pow(inverse_labda,5)
            c_4 = -1.3064 + 0.0163562 * math.pow(inverse_labda,5)
        elif 415 <= labda <= 1100:
            c_0 = 0.7560 - 0.26754 * inverse_labda
            c_2 = -0.0433 + 0.0010059 * math.pow(inverse_labda,5)
            c_3 = 0.2496 - 0.0049131  * math.pow(inverse_labda,5)
            c_4 = -0.1168 + 0.003494 * math.pow(inverse_labda,5)
        else:
            logger.critical('The lambda value does not exist!')
            raise Exception('The lambda value does not exist!')

        intensity = 0
        for r in np.arange(0,1,step = dr):
            mu = np.sqrt(1-math.pow(r,2))   
            L = c_0 + (1-c_0) * mu + c_2 * (mu*(math.log(2/(1+1/mu)))/min_xi_2) + c_3 * (mu*(-math.log(2) + mu * math.log(1+1/mu))/min_xi_3) + c_4 * (mu*(math.log(2) - 1 + mu - math.pow(mu,2)*(math.log(1+1/mu)))/min_xi_4)
            intensity += L * (math.pi * (((r*self.radius)+(dr*self.radius))**2 - (r*self.radius)**2))

        return intensity



class Planet():
    def __init__(self, a, e, radius, mass, theta):
        self.a = a
        self.e = e
        self.radius = radius
        self.radius_sqr2 = math.pow(radius, 2)
        self.mass = mass
        self.theta = theta

        self.b = math.sqrt(math.pow(self.a, 2) - math.pow((self.e * self.a), 2))

    def initialize(self, phi):
        self.phi = phi

    def calcOrbitRadius(self):
        return (self.a * (1 - self.e * self.e)) / (1 + self.e * math.cos(self.phi))

    def updatePos(self):
        self.x = self.calcOrbitRadius() * math.cos(self.phi) + (self.e * self.a)
        self.y = self.calcOrbitRadius() * math.cos(self.theta) * math.sin(self.phi)
        self.z = self.calcOrbitRadius() * math.sin(self.theta) * math.sin(self.phi)


class Moon():
    def __init__(self, radius, orbit_radius, phi, inclination, theta):
        self.radius = radius
        self.radius_sqr2 = math.pow(radius, 2)
        self.orbit_radius = orbit_radius
        self.phi = phi
        self.inclination = inclination
        self.theta = theta


    def updatePos(self):
        #dit klopt nog niet!
        self.x = self.orbit_radius * math.cos(self.phi) * math.sin(self.theta)
        
        #dit klopt nog niet helemaal
        self.y = self.orbit_radius * math.cos(self.phi) * math.cos(self.inclination) * math.cos(self.theta)
        
        self.z = self.orbit_radius * math.sin(self.phi) * math.sin(self.inclination)

