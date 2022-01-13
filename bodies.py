import math

class Star():
  def __init__(self,x,y, radius, mass, temperature):
    self.x = x
    self.y = y
    
    self.radius = radius
    self.radius_sqr2 = math.pow(radius, 2)
    self.mass = mass
    self.temperature = temperature
    
    #This is just the area of a circle
    self.intensity = math.pi * math.pow(radius, 2)

  def foo(self):
    pass

class Planet():
  def __init__(self,x,y, radius, mass, orbit_radius):
    self.x = x
    self.y = y
    self.radius = radius
    self.radius_sqr2 = math.pow(radius, 2)
    self.mass = mass
    self.orbit_radius = orbit_radius

  def foo(self):
    pass