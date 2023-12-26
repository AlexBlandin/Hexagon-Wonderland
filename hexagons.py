from functools import cache
from math import cos, pi, sin, sqrt
from typing import NamedTuple, Self


class Point(NamedTuple):
  """A point in a Cartesian (x,y) space"""

  x: float
  y: float


class ColRow(NamedTuple):
  """A point in a (col,row) space"""

  col: int
  row: int


class DoubledCoord(ColRow):
  """Coordinates in a Doubled (col,row) grid space"""

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


class Offset(ColRow):
  """Coordinates in Offset (col,row) space, the type of offset system is baked in for uniformity"""

  @classmethod
  @property
  def even(cls):
    return 1

  @classmethod
  @property
  def odd(cls):
    return -1

  def qoffset_to_cube(self, offset: int):
    if offset not in {Offset.even, Offset.odd}:
      raise ValueError("offset must be EVEN (+1) or ODD (-1)")  # noqa: TRY003
    q = self.col
    r = self.row - (self.col + offset * (self.col & 1)) // 2
    return Hex(q, r, -q - r)

  def roffset_to_cube(self, offset: int):
    if offset not in {Offset.even, Offset.odd}:
      raise ValueError("offset must be EVEN (+1) or ODD (-1)")  # noqa: TRY003
    q = self.col - (self.row + offset * (self.row & 1)) // 2
    r = self.row
    return Hex(q, r, -q - r)


class Hex(NamedTuple):  # noqa: PLR0904
  """A Hexagon, defined as a cube analogue in the space (q,r,s) where q + r + s == 0"""

  q: int | float
  r: int | float
  s: int | float
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
    return self if n <= 0 else self.rotate_left << (n - 1)

  def __rshift__(self, n: int):
    return self if n <= 0 else self.rotate_right >> (n - 1)

  @staticmethod
  @cache
  def directions():
    return Hex(1, 0, -1), Hex(1, -1, 0), Hex(0, -1, 1), Hex(-1, 0, 1), Hex(-1, 1, 0), Hex(0, 1, -1)

  @staticmethod
  @cache
  def diagonals():
    return Hex(2, -1, -1), Hex(1, -2, 1), Hex(-1, -1, 2), Hex(-2, 1, 1), Hex(-1, 2, -1), Hex(1, 1, -2)

  @staticmethod
  @cache
  def direction(direction: int):
    return Hex.directions()[direction]

  def neighbour(self, direction: int):
    return self + Hex.directions()[direction]

  @property
  def neighbours(self):
    d = Hex.directions()
    return self + d[0], self + d[1], self + d[2], self + d[3], self + d[4], self + d[5]

  def diagonal_neighbour(self, diagonal: int):
    return self + Hex.diagonals()[diagonal]

  @property
  def diagonal_neighbours(self):
    d = Hex.diagonals()
    return self + d[0], self + d[1], self + d[2], self + d[3], self + d[4], self + d[5]

  def __abs__(self):
    return (abs(self.q) + abs(self.r) + abs(self.s)) // 2

  def distance(self, other: Self):
    return abs(self - other)

  def __round__(self):
    qi, ri, si = int(round(self.q)), int(round(self.r)), int(round(self.s))
    q_diff, r_diff, s_diff = abs(qi - self.q), abs(ri - self.r), abs(si - self.s)
    if q_diff > r_diff and q_diff > s_diff:
      qi = -ri - si
    elif r_diff > s_diff:
      ri = -qi - si
    else:
      si = -qi - ri
    return Hex(qi, ri, si)

  def lerp(self, other: Self, t: float):
    return Hex(self.q * (1.0 - t) + other.q * t, self.r * (1.0 - t) + other.r * t, self.s * (1.0 - t) + other.s * t)

  def linedraw(self, other: Self) -> list[Self]:
    N = round(self.distance(other))
    a_nudge = Hex(self.q + 1e-06, self.r + 1e-06, self.s - 2e-06)
    b_nudge = Hex(other.q + 1e-06, other.r + 1e-06, other.s - 2e-06)
    step = 1.0 / max(N, 1)
    return [round(a_nudge.lerp(b_nudge, step * i)) for i in range(N + 1)]  # type: ignore

  @property
  def reflect_q(self):
    return Hex(self.q, self.s, self.r)

  @property
  def reflect_r(self):
    return Hex(self.s, self.r, self.q)

  @property
  def reflect_s(self):
    return Hex(self.r, self.q, self.s)

  def range(self, N: int):
    """Given a range N, which hexes are within N steps from here?"""
    return [self + Hex(q, r, -q - r) for q in range(-N, N + 1) for r in range(max(-N, -q - N), min(+N, -q + N) + 1)]

  def reachable(self, movement: int):
    """Given a number of steps that can be made, which hexes are reachable?"""
    visited: set[Hex] = {self}
    fringes: list[list[Hex]] = []
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

  def ring(self, radius: int):
    """Given a number of steps outwards, which hexes form a ring at this distance?"""
    results: list[Hex] = []
    if radius == 0:
      return [self]
    hex: Hex = self + self.direction(4) * radius
    for i in range(6):
      for _ in range(radius):
        results.append(hex)
        hex = hex.neighbour(i)
    return results

  def spiral(self, radius: int):
    """Given a number of steps outwards, cover the area in that ring in a single spiral"""
    results: list[Hex] = []
    for k in range(radius + 1):
      results += self.ring(k)
    return results  # TODO: speedup? flatten instead of concats

  def qoffset_from_cube(self, offset: int):
    if offset not in {Offset.even, Offset.odd}:
      raise ValueError("offset must be EVEN (+1) or ODD (-1)")  # noqa: TRY003
    hex: Hex = round(self)  # type: ignore
    return Offset(int(hex.q), int(hex.r) + (int(hex.q) + offset * (int(hex.q) & 1)) // 2)

  def roffset_from_cube(self, offset: int):
    if offset not in {Offset.even, Offset.odd}:
      raise ValueError("offset must be EVEN (+1) or ODD (-1)")  # noqa: TRY003
    hex: Hex = round(self)  # type: ignore
    return Offset(int(hex.q) + (int(hex.r) + offset * (int(hex.r) & 1)) // 2, int(hex.r))

  @property
  def qdoubled_from_cube(self):
    hex: Hex = round(self)  # type: ignore
    return DoubledCoord(int(hex.q), 2 * int(hex.r) + int(hex.q))

  @property
  def rdoubled_from_cube(self):
    hex: Hex = round(self)  # type: ignore
    return DoubledCoord(2 * int(hex.q) + int(hex.r), int(hex.r))


class Orientation(NamedTuple):
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


class Layout(NamedTuple):
  """Do you want the hexagons to have pointy tops or flat tops?"""

  orientation: Orientation
  size: Point
  origin: Point

  @classmethod
  @property
  def pointy(cls):
    return Orientation(sqrt(3.0), sqrt(3.0) / 2.0, 0.0, 3.0 / 2.0, sqrt(3.0) / 3.0, -1.0 / 3.0, 0.0, 2.0 / 3.0, 0.5)

  @classmethod
  @property
  def flat(cls):
    return Orientation(3.0 / 2.0, 0.0, sqrt(3.0) / 2.0, sqrt(3.0), 2.0 / 3.0, 0.0, -1.0 / 3.0, sqrt(3.0) / 3.0, 0.0)

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


def equal_rowcol(name: str, a: ColRow, b: ColRow):
  if not (a.col == b.col and a.row == b.row):
    complain(name, a, b)


def equal_any(name: str, a, b):
  if a != b:
    complain(name, a, b)


def equal_hex_array(name: str, a: list[Hex], b: list[Hex]):
  equal_any(name, len(a), len(b))
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
  equal_any("hex_distance", 7, Hex(3, -7, 4).distance(Hex(0, 0, 0)))


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
  equal_hex("hex_round 1", Hex(5, -10, 5), round(Hex(0.0, 0.0, 0.0).lerp(Hex(10.0, -20.0, 10.0), 0.5)))  # type: ignore
  equal_hex("hex_round 2", round(a), round(a.lerp(b, 0.499)))  # type: ignore
  equal_hex("hex_round 3", round(b), round(a.lerp(b, 0.501)))  # type: ignore
  equal_hex(
    "hex_round 4",
    round(a),  # type: ignore
    round(Hex(a.q * 0.4 + b.q * 0.3 + c.q * 0.3, a.r * 0.4 + b.r * 0.3 + c.r * 0.3, a.s * 0.4 + b.s * 0.3 + c.s * 0.3)),  # type: ignore
  )
  equal_hex(
    "hex_round 5",
    round(c),  # type: ignore
    round(Hex(a.q * 0.3 + b.q * 0.3 + c.q * 0.4, a.r * 0.3 + b.r * 0.3 + c.r * 0.4, a.s * 0.3 + b.s * 0.3 + c.s * 0.4)),  # type: ignore
  )


def test_hex_linedraw():
  equal_hex_array(
    "hex_linedraw", [Hex(0, 0, 0), Hex(0, -1, 1), Hex(0, -2, 2), Hex(1, -3, 2), Hex(1, -4, 3), Hex(1, -5, 4)], Hex(0, 0, 0).linedraw(Hex(1, -5, 4))
  )


def test_layout():
  h = Hex(3, 4, -7)
  flat = Layout(Layout.flat, Point(10.0, 15.0), Point(35.0, 71.0))
  equal_hex("layout flat", h, round(flat.pixel_to_hex(flat.hex_to_pixel(h))))  # type: ignore
  pointy = Layout(Layout.pointy, Point(10.0, 15.0), Point(35.0, 71.0))
  equal_hex("layout pointy", h, round(pointy.pixel_to_hex(pointy.hex_to_pixel(h))))  # type: ignore


def test_offset_roundtrip():
  a = Hex(3, 4, -7)
  b = Offset(1, -3)
  equal_hex("roundtrip even-q-from", a, a.qoffset_from_cube(Offset.even).qoffset_to_cube(Offset.even))
  equal_rowcol("roundtrip even-q-to", b, b.qoffset_to_cube(Offset.even).qoffset_from_cube(Offset.even))
  equal_hex("roundtrip odd-q-from-to", a, a.qoffset_from_cube(Offset.odd).qoffset_to_cube(Offset.odd))
  equal_rowcol("roundtrip odd-q-to", b, b.qoffset_to_cube(Offset.odd).qoffset_from_cube(Offset.odd))
  equal_hex("roundtrip even-r-from", a, a.roffset_from_cube(Offset.even).roffset_to_cube(Offset.even))
  equal_rowcol("roundtrip even-r-to", b, b.roffset_to_cube(Offset.even).roffset_from_cube(Offset.even))
  equal_hex("roundtrip odd-r-from", a, a.roffset_from_cube(Offset.odd).roffset_to_cube(Offset.odd))
  equal_rowcol("roundtrip odd-r-to", b, b.roffset_to_cube(Offset.odd).roffset_from_cube(Offset.odd))


def test_offset_from_cube():
  equal_rowcol("offset_from_cube even-q", Offset(1, 3), Hex(1, 2, -3).qoffset_from_cube(Offset.even))
  equal_rowcol("offset_from_cube odd-q", Offset(1, 2), Hex(1, 2, -3).qoffset_from_cube(Offset.odd))


def test_offset_to_cube():
  equal_hex("offset_to_cube even-q", Hex(1, 2, -3), Offset(1, 3).qoffset_to_cube(Offset.even))
  equal_hex("offset_to_cube odd-q", Hex(1, 2, -3), Offset(1, 2).qoffset_to_cube(Offset.odd))


def test_doubled_roundtrip():
  a = Hex(3, 4, -7)
  b = DoubledCoord(1, -3)
  equal_hex("roundtrip doubled-q-from", a, a.qdoubled_from_cube.qdoubled_to_cube)
  equal_rowcol("roundtrip doubled-q-to", b, b.qdoubled_to_cube.qdoubled_from_cube)
  equal_hex("roundtrip doubled-r-from", a, a.rdoubled_from_cube.rdoubled_to_cube)
  equal_rowcol("roundtrip doubled-r-to", b, b.rdoubled_to_cube.rdoubled_from_cube)


def test_doubled_from_cube():
  equal_rowcol("doubled_from_cube doubled-q", DoubledCoord(1, 5), Hex(1, 2, -3).qdoubled_from_cube)
  equal_rowcol("doubled_from_cube doubled-r", DoubledCoord(4, 2), Hex(1, 2, -3).rdoubled_from_cube)


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


if __name__ == "__main__":
  from timeit import timeit

  test_all()
  print("All tests complete.")
  testing_time = timeit("test_all()", number=10**4, globals=globals())
  print(f"{testing_time:.2f}s")
