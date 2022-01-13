import os
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import math
import numpy as np
import matplotlib.pyplot as plt

# import yaml
from tqdm import tqdm

import bodies

# configs


def main():
  simulator = Sim(None, [Star], Planet, [])

  simulator.sim()

  simulator.plot()


class Sim():
  def __init__(self, Config, Stars, Planet, Moons):
    self.C = Config
    self.Stars = Stars  # list of objects
    self.Planet = Planet  # object
    self.Moons = Moons  # list of objects

    self.Nsteps = 1000
    self.Npoints = 150
    
    # determine Theta_start and Theta_stop
    sum_of_radii = 0
    for star in self.Stars:
      sum_of_radii += star.radius  # add stars
    sum_of_radii += self.Planet.radius  # add planet
    for moon in self.Moons:  # add moons
      sum_of_radii += moon.radius
    
    
    self.Theta_start = 2*math.pi - math.acos(-sum_of_radii/self.Planet.orbit_radius)
    self.Theta_stop = 2*math.pi - math.acos(sum_of_radii/self.Planet.orbit_radius)

  def sim(self):
    Stepsize = (self.Theta_stop-self.Theta_start)/self.Nsteps

    self.angles = np.arange(self.Theta_start, (self.Theta_stop + Stepsize), Stepsize)  # x on plot
    self.intensities = []   # y on plot
    
    # CURRENTLY SUPPORTS ONLY 1 STAR!!
    self.surface_per_point =  math.pow(2 * self.Planet.radius, 2) / math.pow(self.Npoints, 2)
    
    for self.Theta in tqdm(self.angles):
      self.step()

    # print(self.intensities)

  def step(self):
    # updating the position of planet
    self.Planet.x = self.Planet.orbit_radius * math.cos(self.Theta)

    # TODO: updating positions of the moons
    
    # overgebleven intensiteit: totale intensiteit - intensiteit achter overlap
    # CURRENTLY SUPPORTS ONLY 1 STAR!!
    thisframe_intensity = self.Stars[0].intensity - self.calcOverlapEffect()
    # print(f'{thisframe_intensity}')
    
    #storing intensity in list
    self.intensities.append(thisframe_intensity)

  def calcOverlapEffect(self):  # S: star, P: planet
    overlapeffect = 0

    # determine difference between x and y values, seen from star POV
    # CURRENTLY SUPPORTS ONLY 1 STAR!!
    dx = self.Stars[0].x - self.Planet.x
    dy = self.Stars[0].y - self.Planet.y

    # check if there is overlap caused by the planet
    
    # print(f'Current edge distance to star edge: {math.sqrt(math.pow(dx,2) + math.pow(dy,2)) - (self.Stars[0].radius + self.Planet.radius)}')
    # CURRENTLY SUPPORTS ONLY 1 STAR!!
    if math.pow(dx,2) + math.pow(dy,2) <= math.pow(self.Stars[0].radius + self.Planet.radius, 2):
      # planet is causing overlap
      # CURRENTLY SUPPORTS ONLY 1 STAR!!
      x_coords = np.arange(self.Planet.x - self.Planet.radius, self.Planet.x + self.Planet.radius, (self.Planet.radius*2)/self.Npoints)
      y_coords = np.arange(self.Planet.y - self.Planet.radius, self.Planet.y + self.Planet.radius, (self.Planet.radius*2)/self.Npoints)

      # make array lists where the x-coords and y-lists are substracted by the planet x- and y-values, to determine distance to planet center. also square them
      x_min_planet_sqr = np.power(x_coords - self.Planet.x, 2)
      y_min_planet_sqr = np.power(y_coords - self.Planet.y, 2)

      # do the same for the star values
      x_min_star_sqr = np.power(x_coords - self.Stars[0].x, 2)
      y_min_star_sqr = np.power(y_coords - self.Stars[0].y, 2)

      for i,x in enumerate(x_coords):
        for j,y in enumerate(y_coords):
          # determine distance from testpoint to middle of the planet and leave it squared
          sqr_distance_to_planet = x_min_planet_sqr[i] + y_min_planet_sqr[j]
          # determine if the point lies within the planet by comparing it against the radius of the planet squared
          if sqr_distance_to_planet <= self.Planet.radius_sqr2:
            # if code reaches here, point is on planet, so determine shade
            # CURRENTLY SUPPORTS ONLY 1 STAR!!
            sqr_distance_to_star = x_min_star_sqr[i] + y_min_star_sqr[j]
            # CURRENTLY SUPPORTS ONLY 1 STAR!!
            if sqr_distance_to_star <= self.Stars[0].radius_sqr2:
              # if code reaches here, point is on on star, so determine shade
              # print(overlapeffect)
              overlapeffect += self.applyLimbDarkening()



    # NOT YET RELEVANT -- FIX WHEN IMPLEMENTING MOONS
    # # check if there is overlap caused by the moons
    # if math.pow(dx,2) + math.pow(dy,2) >= math.pow(S.radius + P.radius, 2):
    #   # planet is causing overlap
    #   surface_per_point =  math.pow(S.radius, 2) / math.pow(self.Npoints, 2)
    #   overlapeffect = 0
    #   for x in np.arange(P.x - P.radius, P.x + P.radius, (P.radius*2)/self.Npoints):
    #     for y in np.arange(P.y - P.radius, P.y + P.radius, (S.radius*2)/self.Npoints)):
    #       sqr_distance_to_planet = math.pow(x-P.x,2) + math.pow(y-P.y,2)
    #       if sqr_distance_to_planet <= math.pow(P.radius, 2):
    #         # if code reaches here, point is on planet
    #         sqr_distance_to_star = math.pow(x-S.x,2) + math.pow(y-S.y,2)
    #         if sqr_distance_to_star <= math.pow(S.radius, 2):
    #           # if code reaches here, point is on on star
    #           overlapeffect += applyLimbDarkening()
    
    return overlapeffect*self.surface_per_point


  def plot(self):
    plt.plot(self.angles, self.intensities)
    plt.show()


  def applyLimbDarkening(self):
    return 1



if __name__ == "__main__":

  Star = bodies.Star(
    x=0,
    y=0,
    radius=696340,  # km
    mass=0,
    temperature = 6000  # K
  )

  Planet = bodies.Planet(
    x=0,
    y=0,
    # radius=6371, #km (10 times the earth)
    radius=63710,
    mass=0,
    orbit_radius = 150000000
  )

  main()