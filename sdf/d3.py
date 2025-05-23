import functools
import numpy as np
# import operator

from . import core, dn, d2, ease

# Constants

ORIGIN = np.array((0, 0, 0))

X = np.array((1, 0, 0))
Y = np.array((0, 1, 0))
Z = np.array((0, 0, 1))

UP = Z

# SDF Class

_ops = {}

class SDF3:
    def __init__(self, f):
        self.f = f
    def __call__(self, p):
        return self.f(p).reshape((-1, 1))
    def __getattr__(self, name):
        if name in _ops:
            return functools.partial(_ops[name], self)
        f = object.__getattribute__(self, 'f')
        if hasattr(f, name):
            return getattr(f, name)
        raise AttributeError(f"'SDF3' object has no attribute '{name}'")
    def __or__(self, other):
        return union(self, other)
    def __and__(self, other):
        return intersection(self, other)
    def __sub__(self, other):
        return difference(self, other)
    def k(self, k=None):
        self._k = k
        return self
    def generate(self, *args, **kwargs):
        return core.generate(self, *args, **kwargs)
    def save(self, path, *args, **kwargs):
        return core.save(path, self, *args, **kwargs)
    def show_slice(self, *args, **kwargs):
        return core.show_slice(self, *args, **kwargs)

def sdf3(f):
    def wrapper(*args, **kwargs):
        return SDF3(f(*args, **kwargs))
    return wrapper

def op3(f):
    def wrapper(*args, **kwargs):
        return SDF3(f(*args, **kwargs))
    _ops[f.__name__] = wrapper
    return wrapper

def op32(f):
    def wrapper(*args, **kwargs):
        return d2.SDF2(f(*args, **kwargs))
    _ops[f.__name__] = wrapper
    return wrapper

# Helpers

def _length(a):
    return np.linalg.norm(a, axis=1)

def _normalize(a):
    return a / np.linalg.norm(a)

def _dot(a, b):
    return np.sum(a * b, axis=1)

def _vec(*arrs):
    return np.stack(arrs, axis=-1)

def _perpendicular(v):
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            return np.cross(v, [0, 1, 0])
    return np.cross(v, [1, 0, 0])

_min = np.minimum
_max = np.maximum

# Primitives

@sdf3
def sphere(radius=1, center=ORIGIN):
    return _sphere(radius=radius, center=center)

class _sphere: 
    def __init__(self, radius, center):
        self.radius = radius
        self.center = center
    def __call__(self, p):
        return _length(p - self.center) - self.radius

@sdf3
def plane(normal=UP, point=ORIGIN):
    return _plane(normal=normal, point=point)

class _plane:
    def __init__(self, normal, point):
        self.normal = _normalize(normal)
        self.point = point
    def __call__(self, p):
        return np.dot(self.point - p, self.normal)

@sdf3
def slab(x0=None, y0=None, z0=None, x1=None, y1=None, z1=None, k=None):
    fs = []
    if x0 is not None:
        fs.append(plane(X, (x0, 0, 0)))
    if x1 is not None:
        fs.append(plane(-X, (x1, 0, 0)))
    if y0 is not None:
        fs.append(plane(Y, (0, y0, 0)))
    if y1 is not None:
        fs.append(plane(-Y, (0, y1, 0)))
    if z0 is not None:
        fs.append(plane(Z, (0, 0, z0)))
    if z1 is not None:
        fs.append(plane(-Z, (0, 0, z1)))
    return intersection(*fs, k=k)


@sdf3
def box(size=1, center=ORIGIN, a=None, b=None):
    return _box(size=size, center=center, a=a, b=b)

class _box:
    def __init__(self, size, center, a, b):
        if a is not None and b is not None:
            self.a = np.array(a)
            self.b = np.array(b)
            self.size = self.b - self.a
            self.center = self.a + self.size / 2 
        if a is None and b is None:
            self.size = np.array(size)
            self.center = center
    def __call__(self, p):
        q = np.abs(p - self.center) - self.size / 2
        return _length(_max(q, 0)) + _min(np.amax(q, axis=1), 0)

@sdf3
def rounded_box(size, radius):
    return _rounded_box(size, radius)

class _rounded_box:
    def __init__(self, size, radius):
        self.size = np.array(size)
        self.radius = radius
    def __call__(self,p):
        q = np.abs(p) - self.size / 2 + self.radius
        return _length(_max(q, 0)) + _min(np.amax(q, axis=1), 0) - self.radius

@sdf3
def wireframe_box(size, thickness):
    return _wireframe_box(size, thickness)

class _wireframe_box:
    def __init__(self, size, thickness):
        self.size = np.array(size)
        self.thickness = thickness
    def g(self, a, b, c):
        return _length(_max(_vec(a, b, c), 0)) + _min(_max(a, _max(b, c)), 0)
    def __call__(self, p):
        p = np.abs(p) - self.size / 2 - self.thickness / 2
        q = np.abs(p + self.thickness / 2) - self.thickness / 2
        px, py, pz = p[:,0], p[:,1], p[:,2]
        qx, qy, qz = q[:,0], q[:,1], q[:,2]
        return _min(_min(self.g(px, qy, qz), self.g(qx, py, qz)), self.g(qx, qy, pz))

@sdf3
def torus(r1, r2):
    return _torus(r1, r2)

class _torus:
    def __init__(self, r1, r2):
        self.r1 = r1
        self.r2 = r2
    def __call__(self, p):
        xy = p[:,[0,1]]
        z = p[:,2]
        a = _length(xy) - self.r1
        b = _length(_vec(a, z)) - self.r2
        return b

@sdf3
def capsule(a, b, radius):
    return _capsule(a, b, radius)

class _capsule:
    def __init__(self, a, b, radius):
        self.a = np.array(a)
        self.b = np.array(b)
        self.radius = radius
    def __call__(self, p):
        pa = p - self.a
        ba = self.b - self.a
        h = np.clip(np.dot(pa, ba) / np.dot(ba, ba), 0, 1).reshape((-1, 1))
        return _length(pa - np.multiply(ba, h)) - self.radius

@sdf3
def cylinder(radius):
    return _cylinder(radius)

class _cylinder:
    def __init__(self, radius):
        self.radius = radius
    def __call__(self, p):
        return _length(p[:,[0,1]]) - self.radius;

@sdf3
def capped_cylinder(a, b, radius):
    return _capped_cylinder(a, b, radius)

class _capped_cylinder:
    def __init__(self, a, b, radius):
        self.a = np.array(a)
        self.b = np.array(b)
        self.radius = radius
    def __call__(self, p):
        ba = self.b - self.a
        pa = p - self.a
        baba = np.dot(ba, ba)
        paba = np.dot(pa, ba).reshape((-1, 1))
        x = _length(pa * baba - ba * paba) - self.radius * baba
        y = np.abs(paba - baba * 0.5) - baba * 0.5
        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))
        x2 = x * x
        y2 = y * y * baba
        d = np.where(
            _max(x, y) < 0,
            -_min(x2, y2),
            np.where(x > 0, x2, 0) + np.where(y > 0, y2, 0))
        return np.sign(d) * np.sqrt(np.abs(d)) / baba

@sdf3
def rounded_cylinder(ra, rb, h):
    return _rounded_cylinder(ra, rb, h)

class _rounded_cylinder:
    def __init__(self, ra, rb, h):
        self.ra = ra
        self.rb = rb
        self.h = h
    def __call__(self, p):   
        d = _vec(
            _length(p[:,[0,1]]) - self.ra + self.rb,
            np.abs(p[:,2]) - self.h / 2 + self.rb)
        return (
            _min(_max(d[:,0], d[:,1]), 0) +
            _length(_max(d, 0)) - self.rb)

@sdf3
def capped_cone(a, b, ra, rb):
    return _capped_cone(a, b, ra, rb)

class _capped_cone:
    def __init__(self, a, b, ra, rb):
        self.a = np.array(a)
        self.b = np.array(b)
        self.ra = ra
        self.rb = rb
    def __call__(self, p):   
        rba = self.rb - self.ra
        baba = np.dot(self.b - self.a, self.b - self.a)
        papa = _dot(p - self.a, p - self.a)
        paba = np.dot(p - self.a, self.b - self.a) / baba
        x = np.sqrt(papa - paba * paba * baba)
        cax = _max(0, x - np.where(paba < 0.5, self.ra, self.rb))
        cay = np.abs(paba - 0.5) - 0.5
        k = rba * rba + baba
        f = np.clip((rba * (x - self.ra) + paba * baba) / k, 0, 1)
        cbx = x - self.ra - f * rba
        cby = paba - f
        s = np.where(np.logical_and(cbx < 0, cay < 0), -1, 1)
        return s * np.sqrt(_min(
            cax * cax + cay * cay * baba,
            cbx * cbx + cby * cby * baba))

@sdf3
def rounded_cone(r1, r2, h):
    return _rounded_cone(r1, r2, h)

class _rounded_cone:
    def __init__(self, r1, r2, h):
        self.r1 = r1
        self.r2 = r2
        self.h = h
    def __call__(self, p): 
        q = _vec(_length(p[:,[0,1]]), p[:,2])
        b = (self.r1 - self.r2) / self.h
        a = np.sqrt(1 - b * b)
        k = np.dot(q, _vec(-b, a))
        c1 = _length(q) - self.r1
        c2 = _length(q - _vec(0, self.h)) - self.r2
        c3 = np.dot(q, _vec(a, b)) - self.r1
        return np.where(k < 0, c1, np.where(k > a * self.h, c2, c3))  

@sdf3
def ellipsoid(size):
    return _ellipsoid(size)

class _ellipsoid:
    def __init__(self, size):
        self.size = np.array(size)
    def __call__(self, p):
        k0 = _length(p / self.size)
        k1 = _length(p / (self.size * self.size))
        return k0 * (k0 - 1) / k1 

@sdf3
def pyramid(h):
    return _pyramid(h)

class _pyramid:
    def __init__(self, h):
        self.h = h
    def __call__(self, p):
        a = np.abs(p[:,[0,1]]) - 0.5
        w = a[:,1] > a[:,0]
        a[w] = a[:,[1,0]][w]
        px = a[:,0]
        py = p[:,2]
        pz = a[:,1]
        m2 = self.h * self.h + 0.25
        qx = pz
        qy = self.h * py - 0.5 * px
        qz = self.h * px + 0.5 * py
        s = _max(-qx, 0)
        t = np.clip((qy - 0.5 * pz) / (m2 + 0.25), 0, 1)
        a = m2 * (qx + s) ** 2 + qy * qy
        b = m2 * (qx + 0.5 * t) ** 2 + (qy - m2 * t) ** 2
        d2 = np.where(
            _min(qy, -qx * m2 - qy * 0.5) > 0,
            0, _min(a, b))
        return np.sqrt((d2 + qz * qz) / m2) * np.sign(_max(qz, -py))

@sdf3
def tetrahedron(r):
    return _tetrahedron(r)

class _tetrahedron:
    def __init__(self, r):
        self.r = r
    def __call__(self, p):   
        x = p[:,0]
        y = p[:,1]
        z = p[:,2]
        return (_max(np.abs(x + y) - z, np.abs(x - y) + z) - self.r) / np.sqrt(3)

@sdf3
def octahedron(r):
    return _octahedron(r)

class _octahedron:
    def __init__(self, r):
        self.r = r
    def __call__(self, p):   
        return (np.sum(np.abs(p), axis=1) - self.r) * np.tan(np.radians(30))

@sdf3
def dodecahedron(r):
    return _dodecahedron(r)

class _dodecahedron:
    def __init__(self, r):
        self.r = r
        self.x, self.y, self.z = _normalize(((1 + np.sqrt(5)) / 2, 1, 0))
    def __call__(self, p):
        p = np.abs(p / self.r)
        a = np.dot(p, (self.x, self.y, self.z))
        b = np.dot(p, (self.z, self.x, self.y))
        c = np.dot(p, (self.y, self.z, self.x))
        q = (_max(_max(a, b), c) - self.x) * self.r
        return q 

@sdf3
def icosahedron(r):
    return _icosahedron(r)

class _icosahedron:
    def __init__(self, r):
        self.r = r * 0.8506507174597755
        self.x, self.y, self.z = _normalize(((np.sqrt(5) + 3) / 2, 1, 0))
        self.w = np.sqrt(3) / 3
    def __call__(self, p):   
        p = np.abs(p / self.r)
        a = np.dot(p, (self.x, self.y, self.z))
        b = np.dot(p, (self.z, self.x, self.y))
        c = np.dot(p, (self.y, self.z, self.x))
        d = np.dot(p, (self.w, self.w, self.w)) - self.x
        return _max(_max(_max(a, b), c) - self.x, d) * self.r

@op3
def translate(other, offset):
    return _translate(other, offset)

class _translate:
    def __init__(self, other, offset):
        self.other = other
        self.offset = offset
    def __call__(self, p):   
        return self.other(p - self.offset)

@op3
def scale(other, factor):
    return _scale(other, factor)

class _scale:
    def __init__(self, other, factor):
        self.other = other
        try:
            self.x, self.y, self.z = factor
        except TypeError:
            self.x = self.y = self.z = factor
        self.s = (self.x, self.y, self.z)
        self.m = min(self.x, min(self.y, self.z))
    def __call__(self, p):   
        return self.other(p / self.s) * self.m

@op3
def rotate(other, angle, vector=Z):
    return _rotate(other, angle, vector=vector)

class _rotate:
    def __init__(self, other, angle, vector):
        self.other = other
        self.x, self.y, self.z = _normalize(vector)
        self.s = np.sin(angle)
        self.c = np.cos(angle)
        self.m = 1 - self.c
        self.matrix = np.array([
            [self.m*self.x*self.x + self.c, self.m*self.x*self.y + self.z*self.s, self.m*self.z*self.x - self.y*self.s],
            [self.m*self.x*self.y - self.z*self.s, self.m*self.y*self.y + self.c, self.m*self.y*self.z + self.x*self.s],
            [self.m*self.z*self.x + self.y*self.s, self.m*self.y*self.z - self.x*self.s, self.m*self.z*self.z + self.c],
        ]).T
    def __call__(self, p):   
        return self.other(np.dot(p, self.matrix))

@op3
def rotate_to(other, a, b):
    a = _normalize(np.array(a))
    b = _normalize(np.array(b))
    dot = np.dot(b, a)
    if dot == 1:
        return other
    if dot == -1:
        return rotate(other, np.pi, _perpendicular(a))
    angle = np.arccos(dot)
    v = _normalize(np.cross(b, a))
    return rotate(other, angle, v)

@op3
def orient(other, axis):
    return rotate_to(other, UP, axis)

@op3
def circular_array(other, count, offset=0):
    return _circular_array(other, count, offset = offset)

class _circular_array:
    def __init__(self, other, count, offset):
        self.other = other.translate(X * offset)
        self.da = 2 * np.pi / count
    def __call__(self, p):   
        x = p[:,0]
        y = p[:,1]
        z = p[:,2]
        d = np.hypot(x, y)
        a = np.arctan2(y, x) % self.da
        d1 = self.other(_vec(np.cos(a - self.da) * d, np.sin(a - self.da) * d, z))
        d2 = self.other(_vec(np.cos(a) * d, np.sin(a) * d, z))
        return _min(d1, d2)

# Alterations

@op3
def elongate(other, size):
    return _elongate(other, size)

class _elongate:
    def __init__(self, other, size):
        self.other = other
        self.size = size
    def __call__(self, p): 
        q = np.abs(p) - self.size
        x = q[:,0].reshape((-1, 1))
        y = q[:,1].reshape((-1, 1))
        z = q[:,2].reshape((-1, 1))
        w = _min(_max(x, _max(y, z)), 0)
        return self.other(_max(q, 0)) + w  

@op3
def twist(other, k):
    return _twist(other, k)

class _twist:
    def __init__(self, other, k):
        self.other = other
        self.k = k
    def __call__(self, p):   
        x = p[:,0]
        y = p[:,1]
        z = p[:,2]
        c = np.cos(self.k * z)
        s = np.sin(self.k * z)
        x2 = c * x - s * y
        y2 = s * x + c * y
        z2 = z
        return self.other(_vec(x2, y2, z2))

@op3
def bend(other, k):
    return _bend(other, k)

class _bend:
    def __init__(self, other, k):
        self.other = other
        self.k = k
    def __call__(self, p):   
        x = p[:,0]
        y = p[:,1]
        z = p[:,2]
        c = np.cos(self.k * x)
        s = np.sin(self.k * x)
        x2 = c * x - s * y
        y2 = s * x + c * y
        z2 = z
        return self.other(_vec(x2, y2, z2))

@op3
def bend_linear(other, p0, p1, v, e=ease.linear):
    return _bend_linear(other, p0, p1, v, e=e)

class _bend_linear:
    def __init__(self, other, p0, p1, v, e):
        self.p0 = np.array(p0)
        self.p1 = np.array(p1)
        self.v = -np.array(v)
        self.e = e
        self.ab = p1 - p0
        self.other = other
    def __call__(self, p):   
        t = np.clip(np.dot(p - self.p0, self.ab) / np.dot(self.ab, self.ab), 0, 1)
        t = self.e(t).reshape((-1, 1))
        return self.other(p + t * self.v)

@op3
def bend_radial(other, r0, r1, dz, e=ease.linear):
    return _bend_radial(other, r0, r1, dz, e=e)

class _bend_radial:
    def __init__(self, other, r0, r1, dz, e):
        self.other = other
        self.r0 = r0
        self.r1 = r1
        self.dz = dz
        self.e = e
    def __call__(self, p): 
        x = p[:,0]
        y = p[:,1]
        z = p[:,2]
        r = np.hypot(x, y)
        t = np.clip((r - self.r0) / (self.r1 - self.r0), 0, 1)
        z = z - self.dz * self.e(t)
        return self.other(_vec(x, y, z))  

@op3
def transition_linear(f0, f1, p0=-Z, p1=Z, e=ease.linear):
    return _transition_linear(f0, f1, p0=p0, p1=p1, e=e)

class _transition_linear:
    def __init__(self, f0, f1, p0, p1, e):
        self.p0 = np.array(p0)
        self.p1 = np.array(p1)
        self.ab = p1 - p0
        self.f0 = f0
        self.f1 = f1
        self.e = e
    def __call__(self, p):   
        d1 = self.f0(p)
        d2 = self.f1(p)
        t = np.clip(np.dot(p - self.p0, self.ab) / np.dot(self.ab, self.ab), 0, 1)
        t = self.e(t).reshape((-1, 1))
        return t * d2 + (1 - t) * d1

@op3
def transition_radial(f0, f1, r0=0, r1=1, e=ease.linear):
    return _transition_radial(f0, f1, r0=r0, r1=r1, e=e)

class _transition_radial:
    def __init__(self, f0, f1, r0, r1, e):
        self.f0 = f0
        self.f1 = f1
        self.r0 = r0
        self.r1 = r1
        self.e = e
    def __call__(self, p):   
        d1 = self.f0(p)
        d2 = self.f1(p)
        r = np.hypot(p[:,0], p[:,1])
        t = np.clip((r - self.r0) / (self.r1 - self.r0), 0, 1)
        t = self.e(t).reshape((-1, 1))
        return t * d2 + (1 - t) * d1

@op3
def wrap_around(other, x0, x1, r=None, e=ease.linear):
    return _wrap_around(other, x0, x1, r=r, e=e)

class _wrap_around:
    def __init__(self, other, x0, x1, r, e):
        self.p0 = X * x0
        self.p1 = X * x1
        self.v = -Y
        self.e = e
        self.other = other
        if r is None:
            self.r = np.linalg.norm(self.p1 - self.p0) / (2 * np.pi)
    def __call__(self, p):   
        x = p[:,0]
        y = p[:,1]
        z = p[:,2]
        d = np.hypot(x, y) - self.r
        d = d.reshape((-1, 1))
        a = np.arctan2(y, x)
        t = (a + np.pi) / (2 * np.pi)
        t = self.e(t).reshape((-1, 1))
        q = self.p0 + (self.p1 - self.p0) * t + self.v * d
        q[:,2] = z
        return self.other(q)

# 3D => 2D Operations

@op32
def slice(other):
    return _slice(other)

class _slice:
    # TODO: support specifying a slice plane
    # TODO: probably a better way to do this
    def __init__(self, other):
        self.other = other
        self.s = slab(z0=-1e-9, z1=1e-9)
        self.a = self.other & self.s
        self.b = self.other.negate() & self.s
    def __call__(self, p):   
        p = _vec(p[:,0], p[:,1], np.zeros(len(p)))
        A = self.a(p).reshape(-1)
        B = -self.b(p).reshape(-1)
        w = A <= 0
        A[w] = B[w]
        return A

# Common

union = op3(dn.union)
difference = op3(dn.difference)
intersection = op3(dn.intersection)
blend = op3(dn.blend)
negate = op3(dn.negate)
dilate = op3(dn.dilate)
erode = op3(dn.erode)
shell = op3(dn.shell)
repeat = op3(dn.repeat)
