from PIL import Image, ImageFont, ImageDraw
import scipy.ndimage as nd
import numpy as np

from . import d2

# TODO: add support for newlines?

PIXELS = 2 ** 22

def _load_image(thing):
    if isinstance(thing, str):
        return Image.open(thing)
    elif isinstance(thing, (np.ndarray, np.generic)):
        return Image.fromarray(thing)
    return Image.fromarray(np.array(thing))

def measure_text(name, text, width=None, height=None):
    font = ImageFont.truetype(name, 96)
    x0, y0, x1, y1 = font.getbbox(text)
    aspect = (x1 - x0) / (y1 - y0)
    if width is None and height is None:
        height = 1
    if width is None:
        width = height * aspect
    if height is None:
        height = width / aspect
    return (width, height)

def measure_image(thing, width=None, height=None):
    im = _load_image(thing)
    w, h = im.size
    aspect = w / h
    if width is None and height is None:
        height = 1
    if width is None:
        width = height * aspect
    if height is None:
        height = width / aspect
    return (width, height)

@d2.sdf2
def text(font_name, text, width=None, height=None, pixels=PIXELS, points=512):
    # load font file
    font = ImageFont.truetype(font_name, points)

    # compute texture bounds
    p = 0.2
    x0, y0, x1, y1 = font.getbbox(text)
    px = int((x1 - x0) * p)
    py = int((y1 - y0) * p)
    tw = x1 - x0 + 1 + px * 2
    th = y1 - y0 + 1 + py * 2

    # render text to image
    im = Image.new('L', (tw, th))
    draw = ImageDraw.Draw(im)
    draw.text((px - x0, py - y0), text, font=font, fill=255)

    return _sdf(width, height, pixels, px, py, im)

@d2.sdf2
def image(thing, width=None, height=None, pixels=PIXELS):
    im = _load_image(thing).convert('L')
    return _sdf(width, height, pixels, 0, 0, im)


def _sdf(width, height, pixels, px, py, im):
    return __sdf(width, height, pixels, px, py, im)

class __sdf:
    def __init__(self, width, height, pixels, px, py, im):
        self.width = width
        self.height = height
        self.pixels = pixels
        self.px = px
        self.py = py
        self.im = im
        self.tw, self.th = self.im.size
        # downscale image if necessary
        factor = (self.pixels / (self.tw * self.th)) ** 0.5
        if factor < 1:
            self.tw, self.th = int(round(self.tw * factor)), int(round(self.th * factor))
            self.px, self.py = int(round(self.px * factor)), int(round(self.py * factor))
            self.im = self.im.resize((self.tw, self.th))

        # convert to numself.py array and apply distance transform
        self.im = self.im.convert('1')
        a = np.array(self.im)
        inside = -nd.distance_transform_edt(a)
        self.outside = nd.distance_transform_edt(~a)
        self.texture = np.zeros(a.shape)
        self.texture[a] = inside[a]
        self.texture[~a] = self.outside[~a]

        # save debug self.image
        # a = np.abs(self.texture)
        # lo, hi = a.min(), a.max()
        # a = (a - lo) / (hi - lo) * 255
        # self.im = Image.fromarray(a.astype('uint8'))
        # self.im.save('debug.png')

        # compute world bounds
        self.pw = self.tw - self.px * 2
        self.ph = self.th - self.py * 2
        aspect = self.pw / self.ph
        if self.width is None and self.height is None:
            self.height = 1
        if self.width is None:
            self.width = self.height * aspect
        if self.height is None:
            self.height = self.width / aspect
        self.x0 = -self.width / 2
        self.y0 = -self.height / 2
        self.x1 = self.width / 2
        self.y1 = self.height / 2

        # scale self.texture distances
        scale = self.width / self.tw
        self.texture *= scale

        # prepare fallback rectangle
        # TODO: reduce size based on mesh resolution instead of dividing by 2
        self.rectangle = d2.rectangle((self.width / 2, self.height / 2))

    def __call__(self, p):
        x = p[:,0]
        y = p[:,1]
        u = (x - self.x0) / (self.x1 - self.x0)
        v = (y - self.y0) / (self.y1 - self.y0)
        v = 1 - v
        i = u * self.pw + self.px
        j = v * self.ph + self.py
        d = _bilinear_interpolate(self.texture, i, j)
        q = self.rectangle(p).reshape(-1)
        self.outside = (i < 0) | (i >= self.tw-1) | (j < 0) | (j >= self.th-1)
        d[self.outside] = q[self.outside]
        return d

def _bilinear_interpolate(a, x, y):
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, a.shape[1] - 1)
    x1 = np.clip(x1, 0, a.shape[1] - 1)
    y0 = np.clip(y0, 0, a.shape[0] - 1)
    y1 = np.clip(y1, 0, a.shape[0] - 1)

    pa = a[y0, x0]
    pb = a[y1, x0]
    pc = a[y0, x1]
    pd = a[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * pa + wb * pb + wc * pc + wd * pd
