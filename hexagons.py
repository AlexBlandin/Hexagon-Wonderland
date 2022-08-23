# Based on http://www.redblobgames.com/grids/hexagons/
from dataclasses import dataclass
from typing import ClassVar
from math import sqrt, sin, cos, pi

@dataclass(slots = True)
class Point:
  x: float
  y: float

@dataclass(slots = True)
class Hex:
  q: float
  r: float
  s: float
  
  def __post_init__(self):
    assert round(self.q + self.r + self.s) == 0, "q + r + s must be 0"
  
  def __add__(self, other: "Hex"):
    return Hex(self.q + other.q, self.r + other.r, self.s + other.s)
  
  def __sub__(self, other: "Hex"):
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
    return self if n == 0 else (self.rotate_left << (n - 1))
  
  def __rshift__(self, n: int):
    return self if n == 0 else (self.rotate_right >> (n - 1))
  
  _directions: ClassVar = [(1, 0, -1), (1, -1, 0), (0, -1, 1), (-1, 0, 1), (-1, 1, 0), (0, 1, -1)]
  _diagonals: ClassVar = [(2, -1, -1), (1, -2, 1), (-1, -1, 2), (-2, 1, 1), (-1, 2, -1), (1, 1, -2)]
  
  @staticmethod
  def direction(direction: int):
    return Hex(*Hex._directions[direction]) # it's faster to do this than load a global
  
  def neighbour(self, direction: int):
    return self + Hex(*Hex._directions[direction])
  
  def diagonal_neighbour(self, diagonal: int):
    return self + Hex(*Hex._diagonals[diagonal])
  
  def __abs__(self):
    return (abs(self.q) + abs(self.r) + abs(self.s)) // 2
  
  def distance(self, other: "Hex"):
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
  
  def lerp(self, other: "Hex", t: float):
    return Hex(self.q * (1.0 - t) + other.q * t, self.r * (1.0 - t) + other.r * t, self.s * (1.0 - t) + other.s * t)
  
  def linedraw(self, other: "Hex"):
    N = self.distance(other)
    a_nudge = Hex(self.q + 1e-06, self.r + 1e-06, self.s - 2e-06)
    b_nudge = Hex(other.q + 1e-06, other.r + 1e-06, other.s - 2e-06)
    step = 1.0 / max(N, 1)
    return [round(a_nudge.lerp(b_nudge, step * i)) for i in range(N + 1)]
  
  def qoffset_from_cube(self, offset):
    if offset != Offset.EVEN and offset != Offset.ODD:
      raise ValueError("offset must be EVEN (+1) or ODD (-1)")
    return Offset(self.q, self.r + (self.q + offset * (self.q & 1)) // 2)
  
  def roffset_from_cube(self, offset):
    if offset != Offset.EVEN and offset != Offset.ODD:
      raise ValueError("offset must be EVEN (+1) or ODD (-1)")
    return Offset(self.q + (self.r + offset * (self.r & 1)) // 2, self.r)
  
  @property
  def qdoubled_from_cube(self):
    return DoubledCoord(self.q, 2 * self.r + self.q)
  
  @property
  def rdoubled_from_cube(self):
    return DoubledCoord(2 * self.q + self.r, self.r)

@dataclass(slots = True)
class Offset:
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

@dataclass(slots = True)
class DoubledCoord:
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

@dataclass(slots = True)
class Orientation:
  f0: float
  f1: float
  f2: float
  f3: float
  b0: float
  b1: float
  b2: float
  b3: float
  start_angle: float

@dataclass(slots = True)
class Layout:
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
    M = self.orientation
    size = self.size
    origin = self.origin
    x = (M.f0 * h.q + M.f1 * h.r) * size.x
    y = (M.f2 * h.q + M.f3 * h.r) * size.y
    return Point(x + origin.x, y + origin.y)
  
  def pixel_to_hex(self, p: Point):
    M = self.orientation
    size = self.size
    origin = self.origin
    pt = Point((p.x - origin.x) / size.x, (p.y - origin.y) / size.y)
    q = M.b0 * pt.x + M.b1 * pt.y
    r = M.b2 * pt.x + M.b3 * pt.y
    return Hex(q, r, -q - r)
  
  def hex_corner_offset(self, corner: int):
    M = self.orientation
    size = self.size
    angle = 2.0 * pi * (M.start_angle - corner) * 0.16666666666666666
    return Point(size.x * cos(angle), size.y * sin(angle))
  
  def polygon_corners(self, h: Hex):
    corners = []
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
  equal_hex("layout", h, round(flat.pixel_to_hex(flat.hex_to_pixel(h))))
  pointy = Layout(Layout.POINTY, Point(10.0, 15.0), Point(35.0, 71.0))
  equal_hex("layout", h, round(pointy.pixel_to_hex(pointy.hex_to_pixel(h))))

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

if __name__ == "__main__":
  test_all()
  print("Done.")
  testing_time = timeit("test_all()", number = 10**5, globals = globals())
  print(f"{testing_time:0.2f}s") # pypy is about 14x faster
