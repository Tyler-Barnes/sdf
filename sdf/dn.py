import itertools
import numpy as np

_min = np.minimum
_max = np.maximum

def union(a, *bs, k=None):
    return _union(a, *bs, k=k)

class _union:
    def __init__(self, a, *bs, k):
        self.a = a
        self.bs = bs
        self.k = k
    def __call__(self, p):
        d1 = self.a(p)
        for b in self.bs:
            d2 = b(p)
            K = self.k or getattr(b, '_k', None)
            if K is None:
                d1 = _min(d1, d2)
            else:
                h = np.clip(0.5 + 0.5 * (d2 - d1) / K, 0, 1)
                m = d2 + (d1 - d2) * h
                d1 = m - K * h * (1 - h)
        return d1

def difference(a, *bs, k=None):
    return _difference(a, *bs, k=k)

class _difference:
    def __init__(self, a, *bs, k):
        self.a = a
        self.bs = bs 
        self.k = k
    def __call__(self, p):
        d1 = self.a(p)
        for b in self.bs:
            d2 = b(p)
            K = self.k or getattr(b, '_k', None)
            if K is None:
                d1 = _max(d1, -d2)
            else:
                h = np.clip(0.5 - 0.5 * (d2 + d1) / K, 0, 1)
                m = d1 + (-d2 - d1) * h
                d1 = m + K * h * (1 - h)
        return d1

def intersection(a, *bs, k=None):
    return _intersection(a,*bs,k=k)

class _intersection:
    def __init__(self,a, *bs, k):
        self.a = a
        self.bs = bs
        self.k = k
    def __call__(self, p):
        d1 = self.a(p)
        for b in self.bs:
            d2 = b(p)
            K = self.k or getattr(b, '_k', None)
            if K is None:
                d1 = _max(d1, d2)
            else:
                h = np.clip(0.5 - 0.5 * (d2 - d1) / K, 0, 1)
                m = d2 + (d1 - d2) * h
                d1 = m + K * h * (1 - h)
        return d1

def blend(a, *bs, k=0.5):
    return _blend(a, *bs, k=k)

class _blend:
    def __init__(self, a, *bs, k):
        self.a = a
        self.bs = bs
        self.k = k
    def __call__(self, p):
        d1 = self.a(p)
        for b in self.bs:
            d2 = b(p)
            K = self.k or getattr(b, '_k', None)
            d1 = K * d2 + (1 - K) * d1
        return d1

def negate(other):
    return _negate(other)

class _negate:
    def __init__(self, other):
        self.other = other
    def __call__(self, p):   
        return -self.other(p)

def dilate(other, r):
    return _dilate(other, r)

class _dilate:
    def __init__(self, other, r):
        self.other = other
        self.r = r
    def __call__(self, p): 
        return self.other(p) - self.r  

def erode(other, r):
    return _erode(other, r)

class _erode:
    def __init__(self, other, r):
        self.other = other
        self.r = r
    def __call__(self, p):   
        return self.other(p) + self.r

def shell(other, thickness):
    return _shell(other, thickness)

class _shell:
    def __init__(self, other, thickness):
        self.other = other
        self.thickness = thickness
    def __call__(self, p):   
        return np.abs(self.other(p)) - self.thickness / 2

def repeat(other, spacing, count=None, padding=0):
    return _repeat(other, spacing, count=count, padding=padding)

class _repeat:
    def __init__(self, other, spacing, count, padding):
        self.count = np.array(count) if count is not None else None
        self.other = other
        self.spacing = np.array(spacing)
        self.padding = padding
    def neighbors(self, dim, padding, spacing):
        try:
            padding = [padding[i] for i in range(dim)]
        except Exception:
            padding = [padding] * dim
        try:
            spacing = [spacing[i] for i in range(dim)]
        except Exception:
            spacing = [spacing] * dim
        for i, s in enumerate(spacing):
            if s == 0:
                padding[i] = 0
        axes = [list(range(-p, p + 1)) for p in padding]
        return list(itertools.product(*axes))
    def __call__(self, p):   
        q = np.divide(p, self.spacing, out=np.zeros_like(p), where=self.spacing != 0)
        if self.count is None:
            index = np.round(q)
        else:
            index = np.clip(np.round(q), -self.count, self.count)

        indexes = [index + n for n in self.neighbors(p.shape[-1], self.padding, self.spacing)]
        A = [self.other(p - self.spacing * i) for i in indexes]
        a = A[0]
        for b in A[1:]:
            a = _min(a, b)
        return a
