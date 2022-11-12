import sys
from dataclasses import dataclass
from functools import cache, partial
from math import cos, pi, sin, sqrt
from typing import ClassVar

# Why this needed to wait until 3.11 for being part of typing, I'll never know
if sys.version_info[0] == 3 and sys.version_info[1] < 11:
  from typing_extensions import Self
else:
  from typing import Self

@dataclass(slots = True, frozen = True)
class Point:
  """A point in Cartesian (x,y) space"""
  x: float
  y: float

@dataclass(slots = True, frozen = True)
class Offset:
  """Coordinates in Offset (col,row) space, the type of offset system is baked in for uniformity"""
  col: float
  row: float
  
  EVEN: ClassVar = 1
  ODD: ClassVar = -1
  
  def qoffset_to_cube(self, offset):
    q = self.col
    r = self.row - (self.col + offset * (self.col & 1)) // 2
    s = -q - r
    if offset != Offset.EVEN and offset != Offset.ODD:
      raise ValueError("offset must be EVEN (+1) or ODD (-1)")
    return Hex(q, r, s)
  
  def roffset_to_cube(self, offset):
    q = self.col - (self.row + offset * (self.row & 1)) // 2
    r = self.row
    s = -q - r
    if offset != Offset.EVEN and offset != Offset.ODD:
      raise ValueError("offset must be EVEN (+1) or ODD (-1)")
    return Hex(q, r, s)

@dataclass(slots = True, frozen = True)
class DoubledCoord:
  """Coordinates in a Doubled (col,row) grid space"""
  col: float
  row: float
  
  @property
  def qdoubled_to_cube(self):
    q = self.col
    r = (self.row - self.col) // 2
    s = -q - r
    return Hex(q, r, s)
  
  @property
  def rdoubled_to_cube(self):
    q = (self.col - self.row) // 2
    r = self.row
    s = -q - r
    return Hex(q, r, s)

@dataclass(slots = True, frozen = True)
class Hex:
  """A Hexagon, defined as a cube analogue in the space (q,r,s) where q + r + s == 0"""
  q: float
  r: float
  s: float
  blocked: bool = False
  
  def __post_init__(self):
    assert round(self.q + self.r + self.s) == 0, "q + r + s must equal 0"
  
  def __add__(self, other: Self):
    return Hex(self.q + other.q, self.r + other.r, self.s + other.s)
  
  def __sub__(self, other: Self):
    return Hex(self.q - other.q, self.r - other.r, self.s - other.s)
  
  def __mul__(self, scale: float):
    return Hex(self.q * scale, self.r * scale, self.s * scale)
  
  def __truediv__(self, scale: float):
    return Hex(self.q / scale, self.r / scale, self.s / scale)
  
  @property
  def rotate_left(self):
    return Hex(-self.s, -self.q, -self.r)
  
  @property
  def rotate_right(self):
    return Hex(-self.r, -self.s, -self.q)
  
  def __lshift__(self, n: int):
    return self if n <= 0 else (self.rotate_left << (n - 1))
  
  def __rshift__(self, n: int):
    return self if n <= 0 else (self.rotate_right >> (n - 1))
  
  @staticmethod
  @cache
  def DIRECTIONS():
    return (Hex(1, 0, -1), Hex(1, -1, 0), Hex(0, -1, 1), Hex(-1, 0, 1), Hex(-1, 1, 0), Hex(0, 1, -1))
  
  @staticmethod
  @cache
  def DIAGONALS():
    return (Hex(2, -1, -1), Hex(1, -2, 1), Hex(-1, -1, 2), Hex(-2, 1, 1), Hex(-1, 2, -1), Hex(1, 1, -2))
  
  @staticmethod
  @cache
  def direction(direction: int):
    return Hex.DIRECTIONS()[direction]
  
  def neighbour(self, direction: int):
    return self + Hex.DIRECTIONS()[direction]
  
  @property
  def neighbours(self):
    d = Hex.DIRECTIONS()
    return (self + d[0], self + d[1], self + d[2], self + d[3], self + d[4], self + d[5])
  
  def diagonal_neighbour(self, diagonal: int):
    return self + Hex.DIAGONALS()[diagonal]
  
  @property
  def diagonal_neighbours(self):
    d = Hex.DIAGONALS()
    return (self + d[0], self + d[1], self + d[2], self + d[3], self + d[4], self + d[5])
  
  def __abs__(self):
    return (abs(self.q) + abs(self.r) + abs(self.s)) // 2
  
  def distance(self, other: Self):
    return abs(self - other)
  
  def __round__(self):
    qi, ri, si = int(round(self.q)), int(round(self.r)), int(round(self.s))
    qd, rd, sd = abs(qi - self.q), abs(ri - self.r), abs(si - self.s)
    if qd > rd and qd > sd:
      qi = -ri - si
    elif rd > sd:
      ri = -qi - si
    else:
      si = -qi - ri
    return Hex(qi, ri, si)
  
  def lerp(self, other: Self, t: float):
    return Hex(self.q * (1.0 - t) + other.q * t, self.r * (1.0 - t) + other.r * t, self.s * (1.0 - t) + other.s * t)
  
  def linedraw(self, other: Self):
    N = self.distance(other)
    a_nudge = Hex(self.q + 1e-06, self.r + 1e-06, self.s - 2e-06)
    b_nudge = Hex(other.q + 1e-06, other.r + 1e-06, other.s - 2e-06)
    step = 1.0 / max(N, 1)
    return [round(a_nudge.lerp(b_nudge, step * i)) for i in range(N + 1)]
  
  @property
  def reflectQ(self):
    return Hex(self.q, self.s, self.r)
  
  @property
  def reflectR(self):
    return Hex(self.s, self.r, self.q)
  
  @property
  def reflectS(self):
    return Hex(self.r, self.q, self.s)
  
  def range(self, N: int):
    """Given a range N, which hexes are within N steps from here?"""
    return [self + Hex(q, r, -q - r) for q in range(-N, N + 1) for r in range(max(-N, -q - N), min(+N, -q + N) + 1)]
  
  def reachable(self, movement: int):
    """Given a number of steps that can be made, which hexes are reachable?"""
    visited: set[Hex] = {self} # set of hexes
    fringes: list[list[Hex]] = [] # array of arrays of hexes
    fringes.append([self])
    
    for k in range(2, movement + 1):
      fringes.append([])
      for hex in fringes[k - 1]:
        for dir in range(6):
          neighbour = hex.neighbour(dir)
          if neighbour not in visited and not neighbour.blocked:
            visited.add(neighbour)
            fringes[k].append(neighbour)
    
    return visited
  
  def scale(self, factor: int):
    return Hex(self.q * factor, self.r * factor, self.s * factor)
  
  def ring(self, radius: int):
    results: list[Hex] = []
    # this code doesn't work for radius == 0; can you see why?
    hex: Hex = self.scale(self.direction(4), radius)
    for i in range(6):
      for _ in range(radius):
        results.append(hex)
        hex = hex.neighbour(i)
    return results
  
  def spiral(self, radius: int):
    results: list[Hex] = [self]
    for k in range(radius + 1):
      results += self.ring(k)
    return results
  
  def qoffset_from_cube(self, offset: Offset):
    if offset != Offset.EVEN and offset != Offset.ODD:
      raise ValueError("offset must be EVEN (+1) or ODD (-1)")
    return Offset(self.q, self.r + (self.q + offset * (self.q & 1)) // 2)
  
  def roffset_from_cube(self, offset: Offset):
    if offset != Offset.EVEN and offset != Offset.ODD:
      raise ValueError("offset must be EVEN (+1) or ODD (-1)")
    return Offset(self.q + (self.r + offset * (self.r & 1)) // 2, self.r)
  
  @property
  def qdoubled_from_cube(self):
    return DoubledCoord(self.q, 2 * self.r + self.q)
  
  @property
  def rdoubled_from_cube(self):
    return DoubledCoord(2 * self.q + self.r, self.r)

@dataclass(slots = True, frozen = True)
class Orientation:
  """A helper POD for Layout"""
  f0: float
  f1: float
  f2: float
  f3: float
  b0: float
  b1: float
  b2: float
  b3: float
  start_angle: float

@dataclass(slots = True, frozen = True)
class Layout:
  """Do you want the hexagons to have pointy tops or flat tops?"""
  orientation: Orientation
  size: Point
  origin: Point
  
  POINTY: ClassVar = Orientation(
    sqrt(3.0),
    sqrt(3.0) / 2.0, 0.0, 3.0 / 2.0,
    sqrt(3.0) / 3.0, -1.0 / 3.0, 0.0, 2.0 / 3.0, 0.5
  )
  FLAT: ClassVar = Orientation(
    3.0 / 2.0, 0.0,
    sqrt(3.0) / 2.0, sqrt(3.0), 2.0 / 3.0, 0.0, -1.0 / 3.0,
    sqrt(3.0) / 3.0, 0.0
  )
  
  def hex_to_pixel(self, h: Hex):
    o, size, origin = self.orientation, self.size, self.origin
    x = (o.f0 * h.q + o.f1 * h.r) * size.x + origin.x
    y = (o.f2 * h.q + o.f3 * h.r) * size.y + origin.y
    return Point(x, y)
  
  def pixel_to_hex(self, p: Point):
    o, size, origin = self.orientation, self.size, self.origin
    pt = Point((p.x - origin.x) / size.x, (p.y - origin.y) / size.y)
    q = o.b0 * pt.x + o.b1 * pt.y
    r = o.b2 * pt.x + o.b3 * pt.y
    return Hex(q, r, -q - r)
  
  def hex_corner_offset(self, corner: int):
    o, size = self.orientation, self.size
    angle = (o.start_angle - corner) * pi * 0.3333333333333333
    return Point(size.x * cos(angle), size.y * sin(angle))
  
  def polygon_corners(self, h: Hex):
    corners: list[Point] = []
    center = self.hex_to_pixel(h)
    for i in range(6):
      offset = self.hex_corner_offset(i)
      corners.append(Point(center.x + offset.x, center.y + offset.y))
    return corners

# Tests
def complain(*args):
  print(*args)

def equal_hex(name: str, a: Hex, b: Hex):
  if not (a.q == b.q and a.s == b.s and a.r == b.r):
    complain(name, a, b)

def equal_offsetcoord(name: str, a: Offset, b: Offset):
  if not (a.col == b.col and a.row == b.row):
    complain(name, a, b)

def equal_doubledcoord(name: str, a: Offset, b: Offset):
  if not (a.col == b.col and a.row == b.row):
    complain(name, a, b)

def equal_int(name: str, a: int, b: int):
  if not (a == b):
    complain(name, a, b)

def equal_hex_array(name: str, a: list[Hex], b: list[Hex]):
  equal_int(name, len(a), len(b))
  for i in range(0, len(a)):
    equal_hex(name, a[i], b[i])

def test_hex_arithmetic():
  equal_hex("hex_add", Hex(4, -10, 6), Hex(1, -3, 2) + Hex(3, -7, 4))
  equal_hex("hex_subtract", Hex(-2, 4, -2), Hex(1, -3, 2) - Hex(3, -7, 4))
  equal_hex("hex_scale", Hex(2, 4, -6), Hex(1, 2, -3) * 2)

def test_hex_direction():
  equal_hex("hex_direction", Hex(0, -1, 1), Hex.direction(2))

def test_hex_neighbor():
  equal_hex("hex_neighbour", Hex(1, -3, 2), Hex(1, -2, 1).neighbour(2))

def test_hex_diagonal():
  equal_hex("hex_diagonal", Hex(-1, -1, 2), Hex(1, -2, 1).diagonal_neighbour(3))

def test_hex_distance():
  equal_int("hex_distance", 7, Hex(3, -7, 4).distance(Hex(0, 0, 0)))

def test_hex_rotate_right():
  equal_hex("hex_rotate_right", Hex(1, -3, 2).rotate_right, Hex(3, -2, -1))
  equal_hex("hex_rotate_right 1", Hex(1, -3, 2) >> 1, Hex(3, -2, -1))
  equal_hex("hex_rotate_right 2", Hex(1, -3, 2) >> 2, Hex(3, -2, -1).rotate_right)

def test_hex_rotate_left():
  equal_hex("hex_rotate_left", Hex(1, -3, 2).rotate_left, Hex(-2, -1, 3))
  equal_hex("hex_rotate_left 1", Hex(1, -3, 2) << 1, Hex(-2, -1, 3))
  equal_hex("hex_rotate_left 2", Hex(1, -3, 2) << 2, Hex(-2, -1, 3).rotate_left)

def test_hex_round():
  a = Hex(0.0, 0.0, 0.0)
  b = Hex(1.0, -1.0, 0.0)
  c = Hex(0.0, -1.0, 1.0)
  equal_hex("hex_round 1", Hex(5, -10, 5), round(Hex(0.0, 0.0, 0.0).lerp(Hex(10.0, -20.0, 10.0), 0.5)))
  equal_hex("hex_round 2", round(a), round(a.lerp(b, 0.499)))
  equal_hex("hex_round 3", round(b), round(a.lerp(b, 0.501)))
  equal_hex(
    "hex_round 4", round(a),
    round(Hex(a.q * 0.4 + b.q * 0.3 + c.q * 0.3, a.r * 0.4 + b.r * 0.3 + c.r * 0.3, a.s * 0.4 + b.s * 0.3 + c.s * 0.3))
  )
  equal_hex(
    "hex_round 5", round(c),
    round(Hex(a.q * 0.3 + b.q * 0.3 + c.q * 0.4, a.r * 0.3 + b.r * 0.3 + c.r * 0.4, a.s * 0.3 + b.s * 0.3 + c.s * 0.4))
  )

def test_hex_linedraw():
  equal_hex_array(
    "hex_linedraw",
    [Hex(0, 0, 0), Hex(0, -1, 1),
     Hex(0, -2, 2), Hex(1, -3, 2),
     Hex(1, -4, 3), Hex(1, -5, 4)],
    Hex(0, 0, 0).linedraw(Hex(1, -5, 4))
  )

def test_layout():
  h = Hex(3, 4, -7)
  flat = Layout(Layout.FLAT, Point(10.0, 15.0), Point(35.0, 71.0))
  equal_hex("layout flat", h, round(flat.pixel_to_hex(flat.hex_to_pixel(h))))
  pointy = Layout(Layout.POINTY, Point(10.0, 15.0), Point(35.0, 71.0))
  equal_hex("layout pointy", h, round(pointy.pixel_to_hex(pointy.hex_to_pixel(h))))

def test_offset_roundtrip():
  a = Hex(3, 4, -7)
  b = Offset(1, -3)
  equal_hex("conversion_roundtrip even-q", a, a.qoffset_from_cube(Offset.EVEN).qoffset_to_cube(Offset.EVEN))
  equal_offsetcoord("conversion_roundtrip even-q", b, b.qoffset_to_cube(Offset.EVEN).qoffset_from_cube(Offset.EVEN))
  equal_hex("conversion_roundtrip odd-q", a, a.qoffset_from_cube(Offset.ODD).qoffset_to_cube(Offset.ODD))
  equal_offsetcoord("conversion_roundtrip odd-q", b, b.qoffset_to_cube(Offset.ODD).qoffset_from_cube(Offset.ODD))
  equal_hex("conversion_roundtrip even-r", a, a.roffset_from_cube(Offset.EVEN).roffset_to_cube(Offset.EVEN))
  equal_offsetcoord("conversion_roundtrip even-r", b, b.roffset_to_cube(Offset.EVEN).roffset_from_cube(Offset.EVEN))
  equal_hex("conversion_roundtrip odd-r", a, a.roffset_from_cube(Offset.ODD).roffset_to_cube(Offset.ODD))
  equal_offsetcoord("conversion_roundtrip odd-r", b, b.roffset_to_cube(Offset.ODD).roffset_from_cube(Offset.ODD))

def test_offset_from_cube():
  equal_offsetcoord("offset_from_cube even-q", Offset(1, 3), Hex(1, 2, -3).qoffset_from_cube(Offset.EVEN))
  equal_offsetcoord("offset_from_cube odd-q", Offset(1, 2), Hex(1, 2, -3).qoffset_from_cube(Offset.ODD))

def test_offset_to_cube():
  equal_hex("offset_to_cube even-", Hex(1, 2, -3), Offset(1, 3).qoffset_to_cube(Offset.EVEN))
  equal_hex("offset_to_cube odd-q", Hex(1, 2, -3), Offset(1, 2).qoffset_to_cube(Offset.ODD))

def test_doubled_roundtrip():
  a = Hex(3, 4, -7)
  b = DoubledCoord(1, -3)
  equal_hex("conversion_roundtrip doubled-q", a, a.qdoubled_from_cube.qdoubled_to_cube)
  equal_doubledcoord("conversion_roundtrip doubled-q", b, b.qdoubled_to_cube.qdoubled_from_cube)
  equal_hex("conversion_roundtrip doubled-r", a, a.rdoubled_from_cube.rdoubled_to_cube)
  equal_doubledcoord("conversion_roundtrip doubled-r", b, b.rdoubled_to_cube.rdoubled_from_cube)

def test_doubled_from_cube():
  equal_doubledcoord("doubled_from_cube doubled-q", DoubledCoord(1, 5), Hex(1, 2, -3).qdoubled_from_cube)
  equal_doubledcoord("doubled_from_cube doubled-r", DoubledCoord(4, 2), Hex(1, 2, -3).rdoubled_from_cube)

def test_doubled_to_cube():
  equal_hex("doubled_to_cube doubled-q", Hex(1, 2, -3), DoubledCoord(1, 5).qdoubled_to_cube)
  equal_hex("doubled_to_cube doubled-r", Hex(1, 2, -3), DoubledCoord(4, 2).rdoubled_to_cube)

def test_all():
  test_hex_arithmetic()
  test_hex_direction()
  test_hex_neighbor()
  test_hex_diagonal()
  test_hex_distance()
  test_hex_rotate_right()
  test_hex_rotate_left()
  test_hex_round()
  test_hex_linedraw()
  test_layout()
  test_offset_roundtrip()
  test_offset_from_cube()
  test_offset_to_cube()
  test_doubled_roundtrip()
  test_doubled_from_cube()
  test_doubled_to_cube()

from timeit import timeit

time = partial(timeit, globals = globals())

if __name__ == "__main__":
  test_all()
  print("All tests complete.")
  testing_time = time("test_all()", number = 10**4)
  print(f"{testing_time:.2f}s") # pypy is about 14x faster
