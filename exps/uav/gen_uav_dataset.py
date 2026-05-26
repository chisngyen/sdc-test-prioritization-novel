"""
UAV-Testing-Competition dataset generator
=========================================
Story: the UAV competition (skhatiri/UAV-Testing-Competition) ships a
*generator framework*, not a labelled test pool. To run our SDC test
prioritization recipe (Transformer + SWA + Focal) on this benchmark we
first need to materialise a labelled pool.

This script samples obstacle placements with the same ranges as the
official RandomGenerator and labels each test as PASS / FAIL.

Modes
-----
- ``--mode surrogate`` (default, NO Docker): label uses a fast geometric
  surrogate -- minimum 3D distance from the planned trajectory to the
  obstacle bounding boxes. ``FAIL`` iff ``min_dist <= --safe_dist``.
  This lets the entire prioritization pipeline run end-to-end without
  the PX4/Aerialist stack. Treat the resulting APFD as ``UAV-surrogate``.
- ``--mode sim`` (real Aerialist): wraps ``snippets/random_generator.py``
  + ``snippets/testcase.py`` and uses the same ``min_distance_to_obstacles``
  metric. Requires Docker + ``aerialist`` Python package installed.

Output
------
``data/uav/uav_dataset_<mode>.json`` with schema::

    [
      {
        "_id": "m1_000000",
        "mission": "mission1",
        "path": [[x,y,z], ...],            # planned trajectory in local ENU
        "obstacles": [
          {"x":..,"y":..,"z":0,"r":..,"l":..,"w":..,"h":..}, ...
        ],
        "min_dist": 1.23,
        "test_outcome": "FAIL" | "PASS"
      },
      ...
    ]

The schema is intentionally compatible with how ``exp_best_all.py``
consumes per-test dicts.
"""
import os, sys, json, math, argparse, random, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
COMP = os.path.join(ROOT, 'external', 'UAV-Testing-Competition', 'snippets')
CASE = os.path.join(COMP, 'case_studies')
OUT  = os.path.join(ROOT, 'data', 'uav')
os.makedirs(OUT, exist_ok=True)

# ----- Generator ranges (mirror snippets/random_generator.py) -----
SIZE_MIN = dict(l=2.0, w=2.0, h=15.0)
SIZE_MAX = dict(l=20.0, w=20.0, h=25.0)
POS_MIN  = dict(x=5.0, y=5.0, r=0.0)
POS_MAX  = dict(x=50.0, y=50.0, r=90.0)

EARTH_R = 6378137.0   # WGS84 equatorial radius (m)


def latlon_to_local(lat, lon, lat0, lon0):
    """Equirectangular projection -> local ENU x (east), y (north) in meters."""
    x = math.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * EARTH_R
    return x, y


def load_mission_path(plan_path):
    """Parse a QGroundControl .plan file -> list of (x, y, z) in local ENU."""
    with open(plan_path, 'r', encoding='utf-8') as f:
        plan = json.load(f)
    items = plan['mission']['items']
    home = plan['mission'].get('plannedHomePosition', None)
    coords = []
    for it in items:
        ps = it.get('params', [])
        if len(ps) >= 7 and ps[4] is not None and ps[5] is not None:
            lat, lon, alt = ps[4], ps[5], ps[6] if ps[6] is not None else 0.0
            coords.append((lat, lon, alt))
    if not coords:
        return np.zeros((0, 3), dtype=np.float64)
    lat0, lon0 = (home[0], home[1]) if home else (coords[0][0], coords[0][1])
    pts = []
    for lat, lon, alt in coords:
        x, y = latlon_to_local(lat, lon, lat0, lon0)
        pts.append([x, y, alt])
    return np.asarray(pts, dtype=np.float64)


def densify(path, step=0.5):
    """Linearly interpolate along ``path`` so segments <= ``step`` meters."""
    if len(path) < 2:
        return path
    out = [path[0]]
    for i in range(1, len(path)):
        a, b = path[i - 1], path[i]
        d = float(np.linalg.norm(b - a))
        n = max(1, int(math.ceil(d / step)))
        for k in range(1, n + 1):
            t = k / n
            out.append(a + (b - a) * t)
    return np.asarray(out, dtype=np.float64)


def sample_obstacle(rng):
    return dict(
        x=rng.uniform(POS_MIN['x'], POS_MAX['x']),
        y=rng.uniform(POS_MIN['y'], POS_MAX['y']),
        z=0.0,
        r=rng.uniform(POS_MIN['r'], POS_MAX['r']),
        l=rng.uniform(SIZE_MIN['l'], SIZE_MAX['l']),
        w=rng.uniform(SIZE_MIN['w'], SIZE_MAX['w']),
        h=rng.uniform(SIZE_MIN['h'], SIZE_MAX['h']),
    )


def point_to_box_dist(p, obs):
    """3D distance from point ``p`` to an axis-aligned-then-rotated box.

    The box is centred at (obs.x, obs.y, obs.z + h/2) with half-extents
    (l/2, w/2, h/2) and yaw ``r`` (degrees) about z. We rotate p into the
    box frame, then compute the standard AABB distance.
    """
    cx, cy, cz = obs['x'], obs['y'], obs['z'] + obs['h'] / 2.0
    hl, hw, hh = obs['l'] / 2.0, obs['w'] / 2.0, obs['h'] / 2.0
    th = math.radians(obs['r'])
    c, s = math.cos(-th), math.sin(-th)
    dx, dy, dz = p[0] - cx, p[1] - cy, p[2] - cz
    rx = dx * c - dy * s
    ry = dx * s + dy * c
    qx = max(0.0, abs(rx) - hl)
    qy = max(0.0, abs(ry) - hw)
    qz = max(0.0, abs(dz) - hh)
    return math.sqrt(qx * qx + qy * qy + qz * qz)


def min_path_obstacle_dist(path_dense, obstacles):
    best = float('inf')
    for p in path_dense:
        for obs in obstacles:
            d = point_to_box_dist(p, obs)
            if d < best:
                best = d
                if best == 0.0:
                    return 0.0
    return best


# --------------------------------------------------------------------
def gen_one(rng, mission_name, path_dense, n_obs_lo, n_obs_hi):
    n = rng.randint(n_obs_lo, n_obs_hi)
    obstacles = [sample_obstacle(rng) for _ in range(n)]
    md = min_path_obstacle_dist(path_dense, obstacles)
    return dict(mission=mission_name, obstacles=obstacles, min_dist=float(md))


def run_surrogate(args, missions):
    rng = random.Random(args.seed)
    out = []
    per_mission = max(1, args.budget // len(missions))
    for mname, path in missions.items():
        path_d = densify(path, step=args.step)
        for i in range(per_mission):
            rec = gen_one(rng, mname, path_d,
                          args.min_obstacles, args.max_obstacles)
            rec['path'] = path.tolist()
            rec['_id'] = f"{mname}_{i:06d}"
            rec['test_outcome'] = 'FAIL' if rec['min_dist'] <= args.safe_dist else 'PASS'
            out.append(rec)
    return out


def run_sim(args, missions):
    """Drive the Aerialist Docker simulator (PyPI aerialist 0.2.1 API).

    Must be invoked with CWD = ``snippets/`` so the relative paths in the
    case_studies/*.yaml resolve. We chdir for the duration of the run.
    """
    import copy as _copy
    saved_cwd = os.getcwd()
    os.chdir(COMP)
    try:
        from aerialist.px4.drone_test import DroneTest, AgentConfig
        from aerialist.px4.obstacle import Obstacle
        from aerialist.px4.docker_agent import DockerAgent
    except ImportError as e:
        os.chdir(saved_cwd)
        sys.exit(f"aerialist not importable: {e}. "
                 f"`conda activate aerialist` (py3.10 env), then retry.")

    rng = random.Random(args.seed)
    out = []
    per_mission = max(1, args.budget // len(missions))
    for mname, path in missions.items():
        base = DroneTest.from_yaml(os.path.join('case_studies', f'{mname}.yaml'))
        for i in range(per_mission):
            obs_dict = sample_obstacle(rng)
            obs = Obstacle(
                Obstacle.Size(l=obs_dict['l'], w=obs_dict['w'], h=obs_dict['h']),
                Obstacle.Position(x=obs_dict['x'], y=obs_dict['y'],
                                  z=obs_dict['z'], r=obs_dict['r']),
            )
            dt = _copy.deepcopy(base)
            dt.simulation.obstacles = [obs]
            if dt.agent is None:
                dt.agent = AgentConfig(engine='docker')
            try:
                agent = DockerAgent(dt)
                results = agent.run()
                if not results:
                    raise RuntimeError("empty results")
                traj = results[0].record
                dists = traj.distance_to_obstacles([obs])
                md = float(min(dists))
            except Exception as e:
                print(f"[sim] skip {mname}_{i}: {e}")
                continue
            print(f"[sim] {mname}_{i:06d} min_dist={md:.2f}")
            out.append(dict(
                _id=f"{mname}_{i:06d}", mission=mname,
                path=path.tolist(), obstacles=[obs_dict],
                min_dist=md,
                test_outcome='FAIL' if md <= args.safe_dist else 'PASS',
            ))
    os.chdir(saved_cwd)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['surrogate', 'sim'], default='surrogate')
    ap.add_argument('--budget', type=int, default=3000,
                    help='Total tests across all missions.')
    ap.add_argument('--missions', nargs='+', default=['mission1', 'mission2', 'mission3'])
    ap.add_argument('--min_obstacles', type=int, default=1)
    ap.add_argument('--max_obstacles', type=int, default=3)
    ap.add_argument('--safe_dist', type=float, default=1.5,
                    help='FAIL iff min distance to any obstacle <= this (m).')
    ap.add_argument('--step', type=float, default=0.5,
                    help='Densification step (m) for the planned trajectory.')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    missions = {}
    for m in args.missions:
        p = os.path.join(CASE, f'{m}.plan')
        if not os.path.exists(p):
            print(f"[warn] missing {p}, skipping")
            continue
        path = load_mission_path(p)
        if len(path) < 2:
            print(f"[warn] {m}.plan has <2 waypoints, skipping")
            continue
        missions[m] = path
        print(f"[mission] {m}: {len(path)} waypoints, span "
              f"x[{path[:,0].min():.1f},{path[:,0].max():.1f}] "
              f"y[{path[:,1].min():.1f},{path[:,1].max():.1f}]")

    if not missions:
        sys.exit("No usable missions; aborting.")

    t0 = time.time()
    print(f"[gen] mode={args.mode} budget={args.budget} safe_dist={args.safe_dist}")
    out = run_surrogate(args, missions) if args.mode == 'surrogate' else run_sim(args, missions)

    n = len(out); n_fail = sum(1 for r in out if r['test_outcome'] == 'FAIL')
    print(f"[gen] {n} tests | FAIL={n_fail} ({100.0*n_fail/max(1,n):.1f}%) | "
          f"{time.time()-t0:.1f}s")

    out_path = os.path.join(OUT, f"uav_dataset_{args.mode}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f)
    print(f"[gen] wrote {out_path}")


if __name__ == '__main__':
    main()
