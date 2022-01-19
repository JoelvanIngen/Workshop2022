import math
import numpy as np


class Star():
    def __init__(self, radius, mass, temperature, labda):
        self.radius = radius
        self.radius_sqr2 = math.pow(radius, 2)
        self.mass = mass
        self.temperature = temperature
        self.labda = labda

    def initialize(self, planet):
        self.x = planet.a * planet.e
        self.y = 0  # for now

        # determine total intensity
        inverse_labda = 1000/self.labda
        dr = 0.001

        min_xi_2 = -0.231961
        min_xi_3 = -0.0772776
        min_xi_4 = -0.0429718

        # list_L = np.zeros(int(1/delta_r))
        # list_r = np.zeros(int(1/delta_r))

        if 303 <= self.labda <= 367:
            c_0 = 0.3998 - 0.10256 * inverse_labda
            c_2 = -0.0063 + 0.0006839 * math.pow(inverse_labda,5)
            c_3 = -0.2291 - 0.0020539 * math.pow(inverse_labda,5)
            c_4 = 0.3324 + 0.001083 * math.pow(inverse_labda,5)
        elif 372 <= self.labda <= 405:
            c_0 = 0.2102 - 0.03570 * inverse_labda
            c_2 = -0.3373 + 0.0043378 * math.pow(inverse_labda,5)
            c_3 = 1.6731 - 0.0206681 * math.pow(inverse_labda,5)
            c_4 = -1.3064 + 0.0163562 * math.pow(inverse_labda,5)
        elif 415 <= self.labda <= 1100:
            c_0 = 0.7560 - 0.26754 * inverse_labda
            c_2 = -0.0433 + 0.0010059 * math.pow(inverse_labda,5)
            c_3 = 0.2496 - 0.0049131  * math.pow(inverse_labda,5)
            c_4 = -0.1168 + 0.003494 * math.pow(inverse_labda,5)

        self.intensity = 0
        for r in np.arange(0,1,step = dr):
            mu = np.sqrt(1-math.pow(r,2))   
            L = c_0 + (1-c_0) * mu + c_2 * (mu*(math.log(2/(1+1/mu)))/min_xi_2) + c_3 * (mu*(-math.log(2) + mu * math.log(1+1/mu))/min_xi_3) + c_4 * (mu*(math.log(2) - 1 + mu - math.pow(mu,2)*(math.log(1+1/mu)))/min_xi_4)
            self.intensity += L * (math.pi * (((r*self.radius)+(dr*self.radius))**2 - (r*self.radius)**2))
        print(self.intensity)
            # list_L[int(r/delta_r)] = L
            # list_r[int(r/delta_r)] = r


        # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # textstr = '\n'.join((r'$\lambda=%.2f$' % (labda, ),))



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

    def updateX(self):
        self.x = self.calcOrbitRadius() * math.cos(self.phi) + (self.e * self.a)

    def updateY(self):
        self.y = self.calcOrbitRadius() * math.cos(self.theta) * math.sin(self.phi)

    def updateZ(self):
        self.z = self.calcOrbitRadius() * math.sin(self.theta) * math.sin(self.phi)
