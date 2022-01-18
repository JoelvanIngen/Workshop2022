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
  'Inclination': math.pi/2,  # pi/2: full transit
  'Number-of-steps': 100,  # number of iteration steps
  'Number-of-points': 150,  # number of integration points per iteration step
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
        temperature = 6000
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


    def addStar(self, radius, mass, temperature):
        self.Star = bodies.Star(radius, mass, temperature)

    def addPlanet(self, a, e, radius, mass):
        self.Planet = bodies.Planet(a, e, radius, mass)

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
        self.Star.initialize(self.Planet, self.applyLimbDarkening)  # determine total intensity etc
        print(self.angles[0])
        self.Planet.initialize(phi = self.angles[0])

        for self.phi in tqdm(self.angles):
            self.doStep()


    def doStep(self):
        # update phi-value of planet
        self.Planet.phi = self.phi

        # determine new x-value for this phi
        self.Planet.updateX()

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
                            overlapeffect += self.applyLimbDarkening()

        return overlapeffect*self.surface_per_point


    def plotResults(self):
        plt.plot(self.angles, self.intensities)
        plt.show()


    def applyLimbDarkening(self):
        return 1



if __name__ == "__main__":

  main()
