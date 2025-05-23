# Writen by Tyler Barnes in 2025
from sdf import *

@sdf3
def mandelbulb(iterations=4, power=8, bailout=3, position=ORIGIN):
    return _mandelbulb(iterations=iterations, power=power, bailout=bailout, position=position)

class _mandelbulb:
    def __init__(self, iterations, power, bailout, position):                         # user parameters
        self.iterations = iterations
        self.power = power
        self.bailout = bailout
        self.position = position
    def spherical(self, zeta):                                                      # cartesian to spherical triplex tranformation 
        r = np.linalg.norm(zeta)                                                    # r = sqrt(x*x+y*y+z*z)
        theta = np.arccos(zeta[2] / r)                                              # θ = acos(z/r) = atan2(sqrt(x*x+y*y), z)
        phi = np.arctan2(zeta[1], zeta[0])                                          # φ = atan2(y/x) = arg(x+yi)
        return r, theta, phi
    def cartesian(self, r, theta, phi):                                             # spherical triplex to cartesian tranformation 
        return np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(phi) * np.sin(theta),
            np.cos(theta)
        ])
    def __call__(self, p):
        _p = p.copy() - self.position                                                              # clone the points object to manipulate 
        _c = _p.copy()                                                              # keep track of the originals 
        dr = np.full(p.shape[0], 1.0)                                               # init the darivitive array 
        r = np.zeros(p.shape[0])                                                    # init the radius array 
        for i in range(p.shape[0]):                                                 # @param i(current iteration)
            zeta = _p[i]                                                            # @param zeta(Z₀: from Z₁ = Z₀ⁿ + C)
            for _ in range(self.iterations):
                r[i], theta, phi = self.spherical(zeta)                             # convert xyz to triplex spherical space 
                if r[i] > self.bailout:                                             # break when point explodes 
                    break
                dr[i] = (r[i] ** (self.power - 1.0)) * self.power * dr[i] + 1.0     # keep track of the expansion of space relative to the object
                zr = r[i] ** self.power         
                theta *= self.power                                                 # Z₀ⁿ: from Z₁ = Z₀ⁿ + C
                phi *= self.power                                                   # Z₀ⁿ: from Z₁ = Z₀ⁿ + C
                zeta = self.cartesian(r, theta, phi) * zr  + _c[i]                  # convert from triplex spherical back to xyz, then add 'C' from Z₁ = Z₀ⁿ + C
            _p[i] = zeta
        d = r / dr                                                                  # ditch the log to play nice with the library
        mask_inside = r < self.bailout
        d[mask_inside] *= -1                                                        # make sure there are actual signs in the signed function 
        return d

LOW  = 100**3                                                                       # draft
MED  = 200**3                                                                       # spot checking
HIGH = 400**3                                                                       # buckle up
if __name__ == "__main__":
    f = mandelbulb().scale(1.5)
    f.save('mandelbulb.stl', samples=MED, sparse=False)
