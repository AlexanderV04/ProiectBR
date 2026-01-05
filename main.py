import random
import heapq
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

# ===== Setări simple =====
WORLD_W, WORLD_H = 20, 20
N_OBS = 6

ROBOT_RADIUS = 1.0        # "cum o fi" (dacă vrei mai "safe", fă 1.0)
CELL = 0.25               # rezoluția grilei (mai mic = mai precis, dar mai lent)
GRID_R, GRID_C = 3, 3      # distribuție obstacole pe hartă (2x3 = răspândite)

# ===== Forme simple =====
def make_square(x, y, s):
    return Polygon([(x,y), (x+s,y), (x+s,y+s), (x,y+s)])

def make_rect(x, y, w, h):
    return Polygon([(x,y), (x+w,y), (x+w,y+h), (x,y+h)])

def make_triangle(x, y, w, h):
    return Polygon([(x,y), (x+w,y), (x+w/2, y+h)])

def generate_obstacles_spread(n_obs):

    obstacles = []
    max_tries = 3000


    min_center_dist = 5.0

    def rand_shape():
        shape = random.choice(["square", "rect", "tri"])

        if shape == "square":
            s = random.uniform(3.0, 4.5)
            x = random.uniform(0.8, WORLD_W - 0.8 - s)
            y = random.uniform(0.8, WORLD_H - 0.8 - s)
            return make_square(x, y, s)

        if shape == "rect":
            w = random.uniform(4.0, 6.0)
            h = random.uniform(3.0, 5.0)
            x = random.uniform(0.8, WORLD_W - 0.8 - w)
            y = random.uniform(0.8, WORLD_H - 0.8 - h)
            return make_rect(x, y, w, h)

        # tri
        w = random.uniform(4.0, 6.0)
        h = random.uniform(3.5, 6.0)
        x = random.uniform(0.8, WORLD_W - 0.8 - w)
        y = random.uniform(0.8, WORLD_H - 0.8 - h)
        return make_triangle(x, y, w, h)

    def center(poly):
        c = poly.centroid
        return (c.x, c.y)

    tries = 0
    while len(obstacles) < n_obs and tries < max_tries:
        tries += 1
        o = rand_shape()
        cx, cy = center(o)

        ok = True
        for q in obstacles:
            qx, qy = center(q)

            # distanță între centre
            if ((cx - qx)**2 + (cy - qy)**2) ** 0.5 < min_center_dist:
                ok = False
                break

            # evită suprapuneri mari
            if o.intersects(q) and o.intersection(q).area > 2.0:
                ok = False
                break

        if ok:
            obstacles.append(o)

    return obstacles

# ===== Grid occupancy din shapely =====
def make_occupancy_grid(expanded_union):
    cols = int(WORLD_W / CELL)
    rows = int(WORLD_H / CELL)
    grid = np.zeros((rows, cols), dtype=np.uint8)

    for r in range(rows):
        y = (r + 0.5) * CELL
        for c in range(cols):
            x = (c + 0.5) * CELL
            if expanded_union.contains(Point(x, y)):
                grid[r, c] = 1
    return grid

def world_to_grid(xy):
    x, y = xy
    r = int(y / CELL)
    c = int(x / CELL)
    return (r, c)

def grid_to_world(rc):
    r, c = rc
    x = (c + 0.5) * CELL
    y = (r + 0.5) * CELL
    return (x, y)

# ===== A* simplu =====
import math
import heapq

def octile(a, b):
    # euristică optimă pentru 8-neigh cu cost 1 și sqrt(2)
    dx = abs(a[1] - b[1])
    dy = abs(a[0] - b[0])
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)

def astar(grid, start, goal):
    rows, cols = grid.shape
    if grid[start] == 1 or grid[goal] == 1:
        return None

    neighbors = [(-1,0),(1,0),(0,-1),(0,1),
                 (-1,-1),(-1,1),(1,-1),(1,1)]

    open_heap = []
    heapq.heappush(open_heap, (0.0, start))
    came_from = {}
    gscore = {start: 0.0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for dr, dc in neighbors:
            nr, nc = current[0] + dr, current[1] + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc] == 1:
                continue

            step_cost = math.sqrt(2) if (dr != 0 and dc != 0) else 1.0
            tentative = gscore[current] + step_cost
            nxt = (nr, nc)

            if tentative < gscore.get(nxt, float("inf")):
                came_from[nxt] = current
                gscore[nxt] = tentative
                f = tentative + octile(nxt, goal)
                heapq.heappush(open_heap, (f, nxt))

    return None

# ===== Desen  =====
def draw(obstacles, start, goal, path_xy):
    fig, ax = plt.subplots(figsize=(8, 8))

    # obstacole:
    for o in obstacles:
        x, y = o.exterior.xy
        ax.fill(x, y, color="dodgerblue", alpha=1.0)

    # start/goal
    ax.plot(start[0], start[1], "go", markersize=10)
    ax.plot(goal[0], goal[1], "ro", markersize=10)

    # traseu
    xs = [p[0] for p in path_xy]
    ys = [p[1] for p in path_xy]
    ax.plot(xs, ys, linewidth=3)  # culoare default, ok

    ax.set_xlim(0, WORLD_W)
    ax.set_ylim(0, WORLD_H)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Traseu găsit")
    plt.show()

def generate_map_with_path(start, goal, max_tries=60):
    """Regenerează obstacole random până găsește drum."""
    for _ in range(max_tries):
        obstacles = generate_obstacles_spread(N_OBS)

        # expandăm pentru raza robotului, DAR nu desenăm expandarea (obstacole rămân simple/albastre)
        expanded_union = unary_union([o.buffer(ROBOT_RADIUS) for o in obstacles])

        # dacă start/goal sunt în zona expandată, regenerează
        if expanded_union.contains(Point(*start)) or expanded_union.contains(Point(*goal)):
            continue

        grid = make_occupancy_grid(expanded_union)

        s_rc = world_to_grid(start)
        g_rc = world_to_grid(goal)

        # clamp
        rows, cols = grid.shape
        s_rc = (min(max(s_rc[0], 0), rows-1), min(max(s_rc[1], 0), cols-1))
        g_rc = (min(max(g_rc[0], 0), rows-1), min(max(g_rc[1], 0), cols-1))

        path_rc = astar(grid, s_rc, g_rc)
        if path_rc:
            path_xy = [grid_to_world(rc) for rc in path_rc]
            return obstacles, path_xy

    return None, None

def main():
    while True:
        print("\n=== START / STOP + Obstacole random (ocolește) ===")
        sx, sy = map(float, input("START (x y) ex: 1 1: ").split())
        gx, gy = map(float, input("STOP  (x y) ex: 19 19: ").split())
        start = (sx, sy)
        goal = (gx, gy)

        obstacles, path_xy = generate_map_with_path(start, goal)

        if obstacles is None:
            print("N-am găsit hartă cu drum (rar). Încearcă alt START/STOP sau rulează din nou.")
        else:
            draw(obstacles, start, goal, path_xy)

        again = input("Regenerezi obstacole random? (y/n): ").strip().lower()
        if again != "y":
            break

if __name__ == "__main__":
    main()
