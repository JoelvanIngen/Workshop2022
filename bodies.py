import math

class Star():
    def __init__(self, radius, mass, temperature):
        self.radius = radius
        self.radius_sqr2 = math.pow(radius, 2)
        self.mass = mass
        self.temperature = temperature


    def initialize(self, planet, limbdarkening):
        self.x = planet.a * planet.e
        self.y = 0  # for now
        # determine total intensity
        # for now, just the area of a circle
        self.intensity = math.pi * math.pow(self.radius, 2)



class Planet():
    def __init__(self,a,e, radius, mass):
        self.a = a
        self.e = e
        self.radius = radius
        self.radius_sqr2 = math.pow(radius, 2)
        self.mass = mass

    def initialize(self, phi):
        self.phi = phi

    def updateX(self):
        r = (self.a * (1 - self.e*self.e)) / (1 + self.e * math.cos(self.phi))
        self.x = r * math.cos(self.phi) + self.e * self.a
        self.y = 0  # for now
