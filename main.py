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
    'lambdas': [None, 1100, 800, 500, 350],  # list of lambdas to use in the simulation
    'Inclination': math.pi/2 + math.pi * 0.00000149,# + 0.0016782 * math.pi,  # pi/2: full transit
    'Number-of-steps': 25,  # number of iteration steps
    'Number-of-points': 25,  # number of integration points per iteration step
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
    )

    sim.addPlanet(
        a = 149597887500,
        e = 0.016710219,
        radius = 6371,
        mass = 0
    )

    # sim.addMoon(
    #     radius = 10000,
    #     orbit_radius = 1000000,
    #     phi=0
    # )

    # sim.addMoon(
    #     radius = 6000,
    #     orbit_radius = 1000000,
    #     phi=math.pi
    # )

    sim.addMoon(
        radius = 1736,
        orbit_radius = .3844e06,
    )
    

    # run simulation
    sim.run()

    # plot results
    sim.plotResults()

class Sim():
    def __init__(self, config):
        self.labdas = config['lambdas']
        self.theta = config['Inclination']
        self.Nsteps = config['Number-of-steps']
        self.Npoints = config['Number-of-points']

        self.moons_added = False  # will be set to true if moons present
        
        logger.debug('Simulation initialized')


    def addStar(self, radius, mass):
        self.Star = bodies.Star(radius, mass)
        logger.info(f'Star added with radius={radius} and mass={mass}')

    def addPlanet(self, a, e, radius, mass):
        self.Planet = bodies.Planet(a, e, radius, mass, self.theta)
        logger.info(f'Planet added with a={a}, e={e}, radius={radius}, mass={mass} and theta={self.theta}')

    def addMoon(self, radius, orbit_radius, phi=0, inclination=math.pi/2, theta=math.pi/2):
        # check if moon list already exists, else make one
        if not self.moons_added:
            self.moons_added = True
            self.Moons = []

        self.Moons.append(bodies.Moon(radius, orbit_radius, phi, inclination, theta))
        logger.info(f'Moon added with radius={radius}, orbit_radius={orbit_radius}, phi={phi}, inclination={inclination} and theta={theta}')

        
    def run(self):
        logger.info('Starting main run')

        # create results list
        self.results = []
        for _ in self.labdas:
            self.results.append([])

        # determine sum of all radii to check starting coordinates
        sum_of_radii = 1.2 * self.Star.radius + self.Planet.radius
        if self.moons_added:
            sum_of_radii += max([moon.orbit_radius + moon.radius for moon in self.Moons])


        # init star
        self.Star.initialize(self.Planet, self.labdas)


        # determine where the planet should start its transit
        x_start = self.Star.x - sum_of_radii
        x_stop = self.Star.x + sum_of_radii
        phi_start = 2*math.pi - math.acos((x_start-self.Star.x)/(self.Planet.a - self.Planet.e*x_start))
        phi_stop = 2*math.pi - math.acos((x_stop-self.Star.x)/(self.Planet.a - self.Planet.e*x_stop))

        logger.debug(f'Phi-start: {phi_start}, phi-stop: {phi_stop}')

        # use steps of constant dPhi
        stepsize = (phi_stop-phi_start)/self.Nsteps

        # create lists to track data
        self.angles = np.arange(phi_start, (phi_stop+stepsize), stepsize)  # x on plot
        self.intensity_per_phi = []

        self.surface_per_point = math.pow(2 * self.Planet.radius, 2) / math.pow(self.Npoints, 2)
        if self.moons_added:
            for moon in self.Moons:
                moon.surface_per_point = math.pow(2 * moon.radius, 2) / math.pow(self.Npoints, 2)

        # initialize start values for the bodies
        self.Planet.initialize(phi = self.angles[0])
        if self.moons_added:
            for moon in self.Moons:
                moon.updatePos()

        # main loop
        for self.phi in tqdm(self.angles):
            self.doStep()

        # normalise brightness list
        


    def doStep(self):
        # update phi-value of planet
        self.Planet.phi = self.phi

        # determine new positions for this phi
        self.Planet.updatePos()

        # overgebleven intensiteit: totale intensiteit - intensiteit achter overlap
        overlapeffect = self.calcOverlapEffectPlanet()

        if self.moons_added:
            overlapeffect += self.calcOverlapEffectMoon()
        intensities_this_frame = np.asarray(self.Star.tot_intensities) - overlapeffect

        #store intensity in list
        self.intensity_per_phi.append(intensities_this_frame)

    def calcOverlapEffectPlanet(self):  # S: star, P: planet
        overlapeffect = np.zeros(len(self.Star.labdas))

        # determine difference between star and planet x and y values, seen from star POV
        dx = self.Star.x - self.Planet.x
        dy = self.Star.y - self.Planet.y

        # check if there is overlap caused by the planet
        if dx*dx + dy*dy > (self.Star.radius + self.Planet.radius)**2: return 0

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
                if sqr_distance_to_planet > self.Planet.radius_sqr2: continue

                # if code reaches here, point is on planet, so determine shade
                sqr_distance_to_star = x_min_star_sqr[i] + y_min_star_sqr[j]
                if sqr_distance_to_star > self.Star.radius_sqr2: continue

                # if code reaches here, point is on on star, so determine shade
                for labda_i,labda in enumerate(self.Star.labdas):
                    if not labda:
                        # no labda
                        overlapeffect[labda_i] += 1
                    else:
                        overlapeffect[labda_i] += self.applyLimbDarkening(sqr_distance_to_star, labda)
        
        return overlapeffect*self.surface_per_point

        # check if there is overlap caused by a moon
    def calcOverlapEffectMoon(self):
        overlapeffect = np.zeros(len(self.Star.labdas))
        for moon_number,moon in enumerate(self.Moons):
            x_relative_to_planet = moon.x
            y_relative_to_planet = moon.y
            x_relative_to_zero = self.Planet.x + x_relative_to_planet
            y_relative_to_zero = self.Planet.y + y_relative_to_planet
            


            # x_relative_to_star = self.Planet.x - moon.x
            # y_relative_to_star = self.Planet.y - moon.y
            dx_star = x_relative_to_zero - self.Star.x
            dy_star = y_relative_to_zero - self.Star.y
            # print(f'moon number {moon_number}', dx_star, dy_star)
        
            # if no overlap with sun, skip
            if dx_star*dx_star + dy_star*dy_star > (self.Star.radius + moon.radius)**2: continue

            # moon is causing overlap
            x_coords = np.arange(x_relative_to_zero - moon.radius, x_relative_to_zero + moon.radius, (moon.radius*2)/self.Npoints)
            # print(x_coords)
            y_coords = np.arange(y_relative_to_zero - moon.radius, y_relative_to_zero + moon.radius, (moon.radius*2)/self.Npoints)
            # print(y_coords)

            # make array lists where the x-coords and y-lists are substracted by the moon x- and y-values, to determine distance to moon center. also square them
            x_min_moon_sqr = np.power(x_coords - self.Planet.x - moon.x, 2)
            y_min_moon_sqr = np.power(y_coords - self.Planet.y - moon.y, 2)

            # do the same for the star values
            x_min_star_sqr = np.power(x_coords - self.Star.x, 2)
            y_min_star_sqr = np.power(y_coords - self.Star.y, 2)

            for i,x in enumerate(x_coords):
                for j,y in enumerate(y_coords):
                    # determine distance from testpoint to middle of the moon and leave it squared
                    sqr_distance_to_moon = x_min_moon_sqr[i] + y_min_moon_sqr[j]
                    # determine if the point lies within the moon by comparing it against the radius of the moon squared
                    # print(sqr_distance_to_moon, moon.radius_sqr2)
                    if sqr_distance_to_moon > moon.radius_sqr2: continue

                    # if code reaches here, point is on moon, so determine shade
                    sqr_distance_to_star = x_min_star_sqr[i] + y_min_star_sqr[j]
                    if sqr_distance_to_star > self.Star.radius_sqr2: continue

                    # if code reaches here, point is on on star

                    # now check if this point isn't overlapping with planet or previous moon
                    # planet first
                    if (x-self.Planet.x)**2 + (y-self.Planet.y)**2 <= self.Planet.radius_sqr2: continue

                    # not hitting planet, now previous moons
                    no_moon_overlap = True
                    for moon_test_i, moon_test in enumerate(self.Moons):
                        if moon_test_i < moon_number:
                            # print(moon_test_i)
                            # print((x-self.Planet.x-moon_test.x)**2 + (y-self.Planet.y-moon_test.y)**2, moon_test.radius_sqr2)
                            if (x-self.Planet.x-moon_test.x)**2 + (y-self.Planet.y-moon_test.y)**2 <= moon_test.radius_sqr2:
                                no_moon_overlap = False
                            
                        # check if value changed, if not, we can calulate shade because point isn't being hit by other body
                    if no_moon_overlap:
                        for labda_i,labda in enumerate(self.Star.labdas):
                            if not labda:
                                # no labda
                                overlapeffect[labda_i] += 1*moon.surface_per_point
                            else:
                                overlapeffect[labda_i] += self.applyLimbDarkening(sqr_distance_to_star, labda)*moon.surface_per_point
                        # overlapeffect += self.applyLimbDarkening(sqr_distance_to_star)*moon.surface_per_point
    

        return overlapeffect

    def applyLimbDarkening(self, sqr_distance_to_star, labda):
        inverse_labda = 1000/labda

        min_xi_2 = -0.231961
        min_xi_3 = -0.0772776
        min_xi_4 = -0.0429718

        # list_L = np.zeros(int(1/delta_r))
        # list_r = np.zeros(int(1/delta_r))

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

        mu = math.sqrt(1-sqr_distance_to_star/self.Star.radius_sqr2)

        L = c_0 + (1-c_0) * mu + c_2 * (mu*(math.log(2/(1+1/mu)))/min_xi_2) + c_3 * (mu*(-math.log(2) + mu * math.log(1+1/mu))/min_xi_3) + c_4 * (mu*(math.log(2) - 1 + mu - mu*mu*(math.log(1+1/mu)))/min_xi_4)

        return L

    def plotResults(self, norm=True):
        # fig, axs = plt.subplots(len(self.labdas))
        # fig.suptitle('waow')
        # for i, labda in enumerate(self.labdas):
        #     axs[i].plot(self.run_results[i][0], self.run_results[i][1])

        # fig.show()
        
        # plt.plot(self.angles, self.intensities)
        # plt.show()
        
        colours = ['k-', 'r--', 'y-.', 'g:', 'b-', 'y--', 'k-.'] 

        intensities_per_labda = [[] for _ in range(len(self.labdas))]
        for i in range(len(self.labdas)):
            for j in range(len(self.intensity_per_phi)):
                intensities_per_labda[i].append(self.intensity_per_phi[j][i])

        # print(intensities_per_labda)
        if norm:
            for i in range(len(intensities_per_labda)):
                intensities_per_labda[i] /= max(intensities_per_labda[i])

        for i, labda in enumerate(self.labdas):
            if not labda:
                plt.plot(self.angles, intensities_per_labda[i], colours[i], label=f'No limb darkening')
            else:
                plt.plot(self.angles, intensities_per_labda[i], colours[i], label=f'$\lambda$={labda} nm')
        plt.legend()
        plt.xlabel("Angle with perihelion [Rad]")
        plt.ylabel("Normalised intensity")
        plt.show()

        # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # textstr = '\n'.join((r'$\lambda=%.2f$' % (labda, ),))
        # plt.text(0.85,0.95, textstr, bbox = props)


if __name__ == "__main__":

    main()
