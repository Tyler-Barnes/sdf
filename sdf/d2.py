import functools
import numpy as np
import operator

from . import dn, d3, ease

# Constants

ORIGIN = np.array((0, 0))

X = np.array((1, 0))
Y = np.array((0, 1))

UP = Y

# SDF Class

_ops = {}

class SDF2:
    def __init__(self, f):
        self.f = f
    def __call__(self, p):
        return self.f(p).reshape((-1, 1))
    def __getattr__(self, name):
        if name in _ops:
            f = _ops[name]
            return functools.partial(f, self)
        raise AttributeError
    def __or__(self, other):
        return union(self, other)
    def __and__(self, other):
        return intersection(self, other)
    def __sub__(self, other):
        return difference(self, other)
    def k(self, k=None):
        self._k = k
        return self

def sdf2(f):
    def wrapper(*args, **kwargs):
        return SDF2(f(*args, **kwargs))
    return wrapper

def op2(f):
    def wrapper(*args, **kwargs):
        return SDF2(f(*args, **kwargs))
    _ops[f.__name__] = wrapper
    return wrapper

def op23(f):
    def wrapper(*args, **kwargs):
        return d3.SDF3(f(*args, **kwargs))
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

_min = np.minimum
_max = np.maximum

# Primitives

@sdf2
def circle(radius=1, center=ORIGIN):
    return _circle(radius=radius, center=center)

class _circle:
    def __init__(self, radius, center):
        self.radius = radius
        self.center = center
    def __call__(self, p):   
        return _length(p - self.center) - self.radius

@sdf2
def line(normal=UP, point=ORIGIN):
    return _line(normal=normal, point=point)

class _line:
    def __init__(self, normal, point):
        self.normal = _normalize(normal)
        self.point = point
    def __call__(self, p):   
        return np.dot(self.point - p, self.normal)

@sdf2
def slab(x0=None, y0=None, x1=None, y1=None, k=None):
    fs = []
    if x0 is not None:
        fs.append(line(X, (x0, 0)))
    if x1 is not None:
        fs.append(line(-X, (x1, 0)))
    if y0 is not None:
        fs.append(line(Y, (0, y0)))
    if y1 is not None:
        fs.append(line(-Y, (0, y1)))
    return intersection(*fs, k=k)

@sdf2
def rectangle(size=1, center=ORIGIN, a=None, b=None):
    return _rectangle(size=size, center=center, a=a, b=b)

class _rectangle:
    def __init__(self, size, center, a, b):
        if a is not None and b is not None:
            self.a = np.array(a)
            self.b = np.array(b)
            self.size = self.b - self.a
            self.center = self.a + self.size / 2
            return rectangle(self.sizesize, self.centercenter)
        self.size = np.array(size)
        self.a = a
        self.b = b
        self.center = center
    def __call__(self, p):   
        q = np.abs(p - self.center) - self.size / 2
        return _length(_max(q, 0)) + _min(np.amax(q, axis=1), 0)

@sdf2
def rounded_rectangle(size, radius, center=ORIGIN):
    return _rounded_rectangle(size, radius, center=center)

class _rounded_rectangle:
    def __init__(self, size, radius, center):
        self.size = size
        self.radius = radius
        self.center = center
        try:
            self.r0, self.r1, self.r2, self.r3 = self.radius
        except TypeError:
            self.r0 = self.r1 = self.r2 = self.r3 = self.radius
    def __call__(self, p):   
        x = p[:,0]
        y = p[:,1]
        r = np.zeros(len(p)).reshape((-1, 1))
        r[np.logical_and(x > 0, y > 0)] = self.r0
        r[np.logical_and(x > 0, y <= 0)] = self.r1
        r[np.logical_and(x <= 0, y <= 0)] = self.r2
        r[np.logical_and(x <= 0, y > 0)] = self.r3
        q = np.abs(p) - self.size / 2 + r
        return (
            _min(_max(q[:,0], q[:,1]), 0).reshape((-1, 1)) +
            _length(_max(q, 0)).reshape((-1, 1)) - r)

@sdf2
def equilateral_triangle():
    return _equilateral_triangle()

class _equilateral_triangle:
    def __init__(self):
        pass
    def __call__(self, p): 
        k = 3 ** 0.5
        p = _vec(
            np.abs(p[:,0]) - 1,
            p[:,1] + 1 / k)
        w = p[:,0] + k * p[:,1] > 0
        q = _vec(
            p[:,0] - k * p[:,1],
            -k * p[:,0] - p[:,1]) / 2
        p = np.where(w.reshape((-1, 1)), q, p)
        p = _vec(
            p[:,0] - np.clip(p[:,0], -2, 0),
            p[:,1])
        return -_length(p) * np.sign(p[:,1])

@sdf2
def hexagon(r):
    return _hexagon(r)

class _hexagon:
    def __init__(self, r):
        self.r = r * 3 ** 0.5 / 2
    def __call__(self, p):   
        k = np.array((3 ** 0.5 / -2, 0.5, np.tan(np.pi / 6)))
        p = np.abs(p)
        p -= 2 * k[:2] * _min(_dot(k[:2], p), 0).reshape((-1, 1))
        p -= _vec(
            np.clip(p[:,0], -k[2] * self.r, k[2] * self.r),
            np.zeros(len(p)) + self.r)
        return _length(p) * np.sign(p[:,1])

@sdf2
def rounded_x(w, r):
    return _rounded_x(w, r)

class _rounded_x:
    def __init__(self, w, r):
        self.w = w
        self.r = r
    def __call__(self, p):   
        p = np.abs(p)
        q = (_min(p[:,0] + p[:,1], self.w) * 0.5).reshape((-1, 1))
        return _length(p - q) - self.r

@sdf2
def polygon(points):
    return _polygon(points)

class _polygon:
    def __init__(self, points):
        self.points = points
    def __call__(self, p):   
        n = len(self.points)
        d = _dot(p - self.points[0], p - self.points[0])
        s = np.ones(len(p))
        for i in range(n):
            j = (i + n - 1) % n
            vi = self.points[i]
            vj = self.points[j]
            e = vj - vi
            w = p - vi
            b = w - e * np.clip(np.dot(w, e) / np.dot(e, e), 0, 1).reshape((-1, 1))
            d = _min(d, _dot(b, b))
            c1 = p[:,1] >= vi[1]
            c2 = p[:,1] < vj[1]
            c3 = e[0] * w[:,1] > e[1] * w[:,0]
            c = _vec(c1, c2, c3)
            s = np.where(np.all(c, axis=1) | np.all(~c, axis=1), -s, s)
        return s * np.sqrt(d)

@sdf2
def vesica(r, d):
    return _vesica(r, d)

class _vesica:
    def __init__(self, r, d):
        self.r = r
        self.d = d
    def __call__(self, p):   
        p = np.abs(p)
        b = np.sqrt(self.r * self.r - self.d * self.d)
        return np.where(
            ((p[:,1] - b) * self.d > p[:,0] * b),
            _length(p - np.array([0, b])),
            _length(p - np.array([-self.d, 0])) - self.r)

# Positioning

@op2
def translate(other, offset):
    return _translate(other, offset)

class _translate:
    def __init__(self, other, offset):
        self.other = other
        self.offset = offset
    def __call__(self, p):
        return self.other(p - self.offset)
@op2
def scale(other, factor):
    return _scale(other, factor)

class _scale:
    def __init__(self, other, factor):
        self.other = other
        self.factor = factor
        try:
            x, y = self.factor
        except TypeError:
            x = y = self.factor
        self.s = (x, y)
        self.m = min(x, y)
    def __call__(self, p):   
        return self.other(p / self.s) * self.m

@op2
def rotate(other, angle):
    return _rotate(other, angle)

class _rotate:
    def __init__(self, other, angle):
        self.other = other
        self.angle = angle
        self.s = np.sin(self.angle)
        self.c = np.cos(self.angle)
        self.m = 1 - self.c
        self.matrix = np.array([
            [self.c, -self.s],
            [self.s, self.c],
        ]).T
    def __call__(self, p): 
        return self.other(np.dot(p, self.matrix))

@op2
def circular_array(other, count):
    return _circular_array(other, count)

class _circular_array:
    def __init__(self, other, count):
        self.other = other
        self.count = count
        self.angles = [i / self.count * 2 * np.pi for i in range(self.count)]
    def __call__(self, p):   
        return union(*[self.other.rotate(a) for a in self.angles])

# Alterations

@op2
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
        w = _min(_max(x, y), 0)
        return self.other(_max(q, 0)) + w

# 2D => 3D Operations

@op23
def extrude(other, h):
    return _extrude(other, h)

class _extrude:
    def __init__(self, other, h):
        self.other = other
        self.h = h
    def __call__(self, p):   
        d = self.other(p[:,[0,1]])
        w = _vec(d.reshape(-1), np.abs(p[:,2]) - self.h / 2)
        return _min(_max(w[:,0], w[:,1]), 0) + _length(_max(w, 0))

@op23
def extrude_to(a, b, h, e=ease.linear):
    return _extrude_to(a, b, h, e=e)

class _extrude_to:
    def __init__(self, a, b, h, e):
        self.a = a
        self.b = b
        self.h = h
        self.e = e
    def __call__(self, p):   
        d1 = self.a(p[:,[0,1]])
        d2 = self.b(p[:,[0,1]])
        t = self.e(np.clip(p[:,2] / self.h, -0.5, 0.5) + 0.5)
        d = d1 + (d2 - d1) * t.reshape((-1, 1))
        w = _vec(d.reshape(-1), np.abs(p[:,2]) - self.h / 2)
        return _min(_max(w[:,0], w[:,1]), 0) + _length(_max(w, 0))

@op23
def revolve(other, offset=0):
    return _revolve(other, offset=offset)

class _revolve:
    def __init__(self, other, offset):
        self.other = other
        self.offset = offset
    def __call__(self, p):   
        xy = p[:,[0,1]]
        q = _vec(_length(xy) - self.offset, p[:,2])
        return self.other(q)

# Common

union = op2(dn.union)
difference = op2(dn.difference)
intersection = op2(dn.intersection)
blend = op2(dn.blend)
negate = op2(dn.negate)
dilate = op2(dn.dilate)
erode = op2(dn.erode)
shell = op2(dn.shell)
repeat = op2(dn.repeat)
