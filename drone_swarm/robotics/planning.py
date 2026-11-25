"""Path planning on a 2D occupancy map of the city: A* (grid, optimal) and RRT* (sampling,
asymptotically optimal), plus path smoothing (shortcutting) and a straight-line baseline."""
import heapq
import numpy as np


def occupancy_from_buildings(buildings, world=(100, 100), res=1.0, inflate=2.0):
    nx, ny = int(world[0] / res), int(world[1] / res)
    xs = (np.arange(nx) + 0.5) * res; ys = (np.arange(ny) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys, indexing='ij')
    occ = np.zeros((nx, ny), bool)
    for b in buildings:
        occ |= (gx - b.x) ** 2 + (gy - b.y) ** 2 < (b.r + inflate) ** 2
    return occ, res


def astar(occ, start, goal, res=1.0):
    s = tuple((np.asarray(start) / res).astype(int)); g = tuple((np.asarray(goal) / res).astype(int))
    nx, ny = occ.shape
    moves = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
    h = lambda a: np.hypot(a[0] - g[0], a[1] - g[1])
    openq = [(h(s), 0.0, s)]; came = {s: None}; cost = {s: 0.0}; expanded = 0
    while openq:
        _, c, cur = heapq.heappop(openq)
        if c > cost[cur]:
            continue
        expanded += 1
        if cur == g:
            break
        for dx, dy, w in moves:
            nb = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nb[0] < nx and 0 <= nb[1] < ny) or occ[nb]:
                continue
            nc = c + w
            if nc < cost.get(nb, np.inf):
                cost[nb] = nc; came[nb] = cur
                heapq.heappush(openq, (nc + h(nb), nc, nb))
    if g not in came:
        return None, expanded
    path = [g]
    while came[path[-1]] is not None:
        path.append(came[path[-1]])
    return (np.array(path[::-1], float) + 0.5) * res, expanded


def _free_segment(occ, res, a, b):
    n = int(np.linalg.norm(b - a) / (res * 0.5)) + 2
    pts = a + (b - a) * np.linspace(0, 1, n)[:, None]
    idx = (pts / res).astype(int)
    idx = np.clip(idx, 0, np.array(occ.shape) - 1)
    return not occ[idx[:, 0], idx[:, 1]].any()


def rrt_star(occ, start, goal, res=1.0, iters=3000, step=6.0, radius=12.0, goal_bias=0.1, seed=0):
    rng = np.random.default_rng(seed)
    W = np.array(occ.shape) * res
    nodes = [np.asarray(start, float)]; parent = [-1]; cost = [0.0]
    goal = np.asarray(goal, float); best = None
    for _ in range(iters):
        q = goal if rng.random() < goal_bias else rng.uniform(0, W)
        P = np.array(nodes)
        i = np.argmin(((P - q) ** 2).sum(1))
        d = q - P[i]; dist = np.linalg.norm(d)
        new = P[i] + d / dist * min(step, dist) if dist > 0 else P[i]
        if not _free_segment(occ, res, P[i], new):
            continue
        near = np.where(((P - new) ** 2).sum(1) < radius ** 2)[0]
        bi, bc = i, cost[i] + np.linalg.norm(new - P[i])
        for j in near:                                         # choose best parent
            c = cost[j] + np.linalg.norm(new - P[j])
            if c < bc and _free_segment(occ, res, P[j], new):
                bi, bc = j, c
        nodes.append(new); parent.append(bi); cost.append(bc); k = len(nodes) - 1
        for j in near:                                         # rewire
            c = bc + np.linalg.norm(P[j] - new)
            if c < cost[j] and _free_segment(occ, res, new, P[j]):
                parent[j], cost[j] = k, c
        if np.linalg.norm(new - goal) < step and _free_segment(occ, res, new, goal):
            c = bc + np.linalg.norm(goal - new)
            if best is None or c < best[1]:
                best = (k, c)
    if best is None:
        return None, np.array(nodes), parent
    path = [goal]; k = best[0]
    while k != -1:
        path.append(nodes[k]); k = parent[k]
    return np.array(path[::-1]), np.array(nodes), parent


def shortcut(path, occ, res=1.0, iters=200, seed=0):
    rng = np.random.default_rng(seed)
    p = [q for q in path]
    for _ in range(iters):
        if len(p) < 3:
            break
        i, j = sorted(rng.choice(len(p), 2, replace=False))
        if j - i > 1 and _free_segment(occ, res, p[i], p[j]):
            p = p[:i + 1] + p[j:]
    return np.array(p)


def path_length(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
