import os
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import math
import numpy as np
import matplotlib.pyplot as plt

# import yaml
from tqdm import tqdm

import bodies

# config
config = {
    'Inclination': math.pi/2 + 0.0016782 * math.pi,  # pi/2: full transit
    'Number-of-steps': 100,  # number of iteration steps
    'Number-of-points': 300,  # number of integration points per iteration step
}

# Constants
G = 6.674e-11

def main():
    # introduce simulation object and set start_values
    sim = Sim(config)

    # set star, planet and add moons
    sim.addStar(
        radius = 696340,
        mass = 1.988e30,
        temperature = 6000,
        labda = 350
    )

    sim.addPlanet(
        a = 150000000,
        e = .2,
        radius = 63710,
        mass = 0
    )



    # run simulation
    sim.run()


    # plot results
    sim.plotResults()


class Sim():
    def __init__(self, config):
        self.theta = config['Inclination']
        self.Nsteps = config['Number-of-steps']
        self.Npoints = config['Number-of-points']

        self.moons_added = False  # will be set to true if moons present


    def addStar(self, radius, mass, temperature, labda):
        self.Star = bodies.Star(radius, mass, temperature, labda)

    def addPlanet(self, a, e, radius, mass):
        self.Planet = bodies.Planet(a, e, radius, mass, self.theta)

    def addMoon(orbit_radius, radius):
        # check if moon list already exists, else make one
        if not self.moons_added:
            self.Moons = []

        self.Moons.append(bodies.Moon(orbit_radius, radius))


    def run(self):
        # determine sum of all radii to check starting coordinates
        sum_of_radii = self.Star.radius + self.Planet.radius
        if self.moons_added:
            sum_of_radii += [moon.radius for moon in self.Moons]


        # determine distance star from zero
        d_star_from_zero = self.Planet.e * self.Planet.a

        # determine where the planet should start its transit
        x_start = d_star_from_zero - sum_of_radii
        x_stop = d_star_from_zero + sum_of_radii
        phi_start = math.pi - math.acos((x_start-d_star_from_zero)/(self.Planet.a - self.Planet.e*x_start))
        phi_stop = math.pi - math.acos((x_stop-d_star_from_zero)/(self.Planet.a - self.Planet.e*x_stop))

        # use steps of constant dPhi
        stepsize = (phi_stop-phi_start)/self.Nsteps

        # create list of phi values and the correspending intensities
        self.angles = np.arange(phi_start, (phi_stop+stepsize), stepsize)  # x on plot
        self.intensities = []  # y on plot

        self.surface_per_point =  math.pow(2 * self.Planet.radius, 2) / math.pow(self.Npoints, 2)

        # initialize start values for the bodies
        self.Star.initialize(self.Planet)  # determine total intensity etc
        self.Planet.initialize(phi = self.angles[0])

        for self.phi in tqdm(self.angles):
            self.doStep()


    def doStep(self):
        # update phi-value of planet
        self.Planet.phi = self.phi

        # determine new positions for this phi
        self.Planet.updateX()
        self.Planet.updateY()

        # overgebleven intensiteit: totale intensiteit - intensiteit achter overlap
        intensity_this_frame = self.Star.intensity - self.calcOverlapEffect()

        #store intensity in list
        self.intensities.append(intensity_this_frame)

    def calcOverlapEffect(self):  # S: star, P: planet
        overlapeffect = 0

        # determine difference between star and planet x and y values, seen from star POV
        dx = self.Star.x - self.Planet.x
        dy = self.Star.y - self.Planet.y

        # check if there is overlap caused by the planet
        if math.pow(dx,2) + math.pow(dy,2) <= math.pow(self.Star.radius + self.Planet.radius, 2):
            # planet is causing overlap
            x_coords = np.arange(self.Planet.x - self.Planet.radius, self.Planet.x + self.Planet.radius, (self.Planet.radius*2)/self.Npoints)
            y_coords = np.arange(self.Planet.y - self.Planet.radius, self.Planet.y + self.Planet.radius, (self.Planet.radius*2)/self.Npoints)

            # make array lists where the x-coords and y-lists are substracted by the planet x- and y-values, to determine distance to planet center. also square them
            x_min_planet_sqr = np.power(x_coords - self.Planet.x, 2)
            y_min_planet_sqr = np.power(y_coords - self.Planet.y, 2)

            # do the same for the star values
            x_min_star_sqr = np.power(x_coords - self.Star.x, 2)
            y_min_star_sqr = np.power(y_coords - self.Star.y, 2)

            for i,x in enumerate(x_coords):
                for j,y in enumerate(y_coords):
                    # determine distance from testpoint to middle of the planet and leave it squared
                    sqr_distance_to_planet = x_min_planet_sqr[i] + y_min_planet_sqr[j]
                    # determine if the point lies within the planet by comparing it against the radius of the planet squared
                    if sqr_distance_to_planet <= self.Planet.radius_sqr2:
                        # if code reaches here, point is on planet, so determine shade
                        sqr_distance_to_star = x_min_star_sqr[i] + y_min_star_sqr[j]
                        if sqr_distance_to_star <= self.Star.radius_sqr2:
                            # if code reaches here, point is on on star, so determine shade
                            overlapeffect += self.applyLimbDarkening(sqr_distance_to_star)

        return overlapeffect*self.surface_per_point


    def plotResults(self):
        plt.plot(self.angles, self.intensities)
        plt.show()


    # def applyLimbDarkening(self, sqr_distance_to_star):
    #     # labda = 1000.67
    #     inverse_labda = 1000/self.Star.labda
    #     # labda_formula = 1000/labda
    #     # delta_r = 0.001

    #     min_xi_2 = -0.231961
    #     min_xi_3 = -0.0772776

    #     # list_L = np.zeros(int(1/delta_r))
    #     # list_r = np.zeros(int(1/delta_r))

    #     if 303 <= self.Star.labda <= 367:
    #         b_0 = 0.3721 - 0.08550 * inverse_labda
    #         b_2 = -0.3371 + 0.07246 * inverse_labda
    #         b_3 = 0.4243 - 0.07246 * inverse_labda - 0.0003209 * math.pow(inverse_labda,5)
    #     elif 372 <= self.Star.labda <= 405:
    #         b_0 = -0.1327 + 0.10590 * inverse_labda
    #         b_2 = 1.8342 - 0.75395 * inverse_labda
    #         b_3 = -1.7650 + 0.75395 * inverse_labda - 0.0004729 * math.pow(inverse_labda,5)
    #     elif 415 <= self.Star.labda <= 1100:
    #         b_0 = 0.7421 - 0.26003 * inverse_labda
    #         b_2 = 0.0644 - 0.04040 * inverse_labda 
    #         b_3 = 0.0286 + 0.04040 * inverse_labda - 0.0005199 * math.pow(inverse_labda,5)

    #     # for r in np.arange(0,1,step = delta_r):
    #     #     mu = np.sqrt(1-math.pow(r,2))
    #     #     L = b_0 + (1-b_0) * mu + b_2 * (mu*(np.log(2) - np.log(1+1/mu))/min_xi_2) + b_3 * (mu*(-np.log(2) + mu* np.log(1+1/mu))/min_xi_3)
    #     #     list_L[int(r/delta_r)] = L
    #     #     list_r[int(r/delta_r)] = r

    #     # plt.plot(list_r, list_L)
    #     # plt.xlabel('Relative radius')
    #     # plt.ylabel('Relative intensity')

    #     # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    #     # textstr = '\n'.join((r'$\lambda=%.2f$' % (labda, ),))
    #     # plt.text(0.85,0.95, textstr, bbox = props)
    #     # plt.show()

    #     mu = math.sqrt(1-sqr_distance_to_star/self.Star.radius_sqr2)

    #     # L = b_0 + (1-b_0) * mu + b_2 * (mu*(math.log(2) - math.log(1+1/mu))/min_xi_2) + b_3 * (mu*(-math.log(2) + mu * math.log(1+1/mu))/min_xi_3)
    #     L = b_0 + (1-b_0) * mu + b_2 * (mu*(math.log(2/(1+1/mu)))/min_xi_2) + b_3 * (mu*(-math.log(2) + mu * math.log(1+1/mu))/min_xi_3)

    #     return L

    def applyLimbDarkening(self, sqr_distance_to_star):
        # labda = 1000.67
        inverse_labda = 1000/self.Star.labda
        # labda_formula = 1000/labda
        # delta_r = 0.001

        min_xi_2 = -0.231961
        min_xi_3 = -0.0772776
        min_xi_4 = -0.0429718

        # list_L = np.zeros(int(1/delta_r))
        # list_r = np.zeros(int(1/delta_r))

        if 303 <= self.Star.labda <= 367:
            c_0 = 0.3998 - 0.10256 * inverse_labda
            c_2 = -0.0063 + 0.0006839 * math.pow(inverse_labda,5)
            c_3 = -0.2291 - 0.0020539 * math.pow(inverse_labda,5)
            c_4 = 0.3324 + 0.001083 * math.pow(inverse_labda,5)
        elif 372 <= self.Star.labda <= 405:
            c_0 = 0.2102 - 0.03570 * inverse_labda
            c_2 = -0.3373 + 0.0043378 * math.pow(inverse_labda,5)
            c_3 = 1.6731 - 0.0206681 * math.pow(inverse_labda,5)
            c_4 = -1.3064 + 0.0163562 * math.pow(inverse_labda,5)
        elif 415 <= self.Star.labda <= 1100:
            c_0 = 0.7560 - 0.26754 * inverse_labda
            c_2 = -0.0433 + 0.0010059 * math.pow(inverse_labda,5)
            c_3 = 0.2496 - 0.0049131  * math.pow(inverse_labda,5)
            c_4 = -0.1168 + 0.003494 * math.pow(inverse_labda,5)

        # for r in np.arange(0,1,step = delta_r):
        #     mu = np.sqrt(1-math.pow(r,2))
        #     L = b_0 + (1-b_0) * mu + b_2 * (mu*(np.log(2) - np.log(1+1/mu))/min_xi_2) + b_3 * (mu*(-np.log(2) + mu* np.log(1+1/mu))/min_xi_3)
        #     list_L[int(r/delta_r)] = L
        #     list_r[int(r/delta_r)] = r

        # plt.plot(list_r, list_L)
        # plt.xlabel('Relative radius')
        # plt.ylabel('Relative intensity')

        # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # textstr = '\n'.join((r'$\lambda=%.2f$' % (labda, ),))
        # plt.text(0.85,0.95, textstr, bbox = props)
        # plt.show()

        mu = math.sqrt(1-sqr_distance_to_star/self.Star.radius_sqr2)

        L = c_0 + (1-c_0) * mu + c_2 * (mu*(math.log(2/(1+1/mu)))/min_xi_2) + c_3 * (mu*(-math.log(2) + mu * math.log(1+1/mu))/min_xi_3) + c_4 * (mu*(math.log(2) - 1 + mu - math.pow(mu,2)*(math.log(1+1/mu)))/min_xi_4)

        return L


if __name__ == "__main__":

  main()
