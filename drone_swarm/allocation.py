"""Multi-robot task allocation: N drones must visit M > N inspection points.

* auction_allocation : decentralised sequential auction (each round every drone bids the
                       marginal cost of inserting each unassigned task into its route; the
                       lowest bid wins) - the core idea of CBBA / SSI auctions.
* greedy_nearest     : each task goes to the drone currently closest to it.
* random_allocation  : baseline.
Routes are improved with 2-opt.  Objective: makespan (longest route) and total distance.
"""
import numpy as np


def route_length(start, pts):
    if len(pts) == 0:
        return 0.0
    path = np.vstack([start, pts])
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def _best_insertion(start, route, task, tasks):
    pts = tasks[route] if route else np.zeros((0, tasks.shape[1]))
    base = route_length(start, pts)
    best = (np.inf, 0)
    for k in range(len(route) + 1):
        new = np.insert(pts, k, tasks[task], axis=0)
        c = route_length(start, new) - base
        if c < best[0]:
            best = (c, k)
    return best


def two_opt(start, route, tasks):
    route = list(route)
    improved = True
    while improved and len(route) > 2:
        improved = False
        for i in range(len(route) - 1):
            for j in range(i + 2, len(route) + 1):
                new = route[:i] + route[i:j][::-1] + route[j:]
                if route_length(start, tasks[new]) < route_length(start, tasks[route]) - 1e-9:
                    route, improved = new, True
    return route


def auction_allocation(starts, tasks, balance=1.0):
    """balance > 0 adds a penalty on the bidder's current route length -> shorter makespan."""
    n = len(starts)
    routes = [[] for _ in range(n)]
    free = set(range(len(tasks)))
    while free:
        best = (np.inf, None, None, None)
        for i in range(n):
            cur = route_length(starts[i], tasks[routes[i]]) if routes[i] else 0.0
            for t in free:
                c, k = _best_insertion(starts[i], routes[i], t, tasks)
                bid = c + balance * cur
                if bid < best[0]:
                    best = (bid, i, t, k)
        _, i, t, k = best
        routes[i].insert(k, t)
        free.remove(t)
    return [two_opt(starts[i], r, tasks) for i, r in enumerate(routes)]


def greedy_nearest(starts, tasks):
    owner = np.linalg.norm(tasks[:, None] - starts[None], axis=-1).argmin(1)
    routes = []
    for i in range(len(starts)):
        r = list(np.where(owner == i)[0])
        r.sort(key=lambda t: np.linalg.norm(tasks[t] - starts[i]))
        routes.append(two_opt(starts[i], r, tasks))
    return routes


def random_allocation(starts, tasks, seed=0):
    rng = np.random.default_rng(seed)
    owner = rng.integers(0, len(starts), len(tasks))
    return [two_opt(starts[i], list(np.where(owner == i)[0]), tasks) for i in range(len(starts))]


def evaluate(starts, tasks, routes):
    L = [route_length(starts[i], tasks[r]) for i, r in enumerate(routes)]
    return dict(makespan=max(L), total=sum(L), tasks_done=sum(len(r) for r in routes))
