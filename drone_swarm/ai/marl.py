"""Multi-agent reinforcement learning for swarm navigation (MAPPO).

Task ("goal swap"): N drones start on a circle and must reach the antipodal point while
avoiding each other and random obstacles in the middle - the classic hardest case for
reactive controllers because every path crosses the centre.

* SwarmNavEnv      : batched, fully vectorised PyTorch environment (B envs x N agents)
* AttentionPolicy  : decentralised actor - each drone attends over its k nearest neighbours
                     and m nearest obstacles (works for any swarm size)
                     centralised critic - also sees a mean-pooled embedding of the whole swarm
                     (Centralised Training, Decentralised Execution = MAPPO)
* train_mappo      : PPO with GAE, clipping, entropy bonus, advantage normalisation
* PotentialFieldController : hand-designed baseline (goal attraction + repulsion)
"""
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- environment
class SwarmNavEnv:
    def __init__(self, n_agents=8, batch=32, arena=40.0, dt=0.2, max_steps=110, k_nb=5, m_obs=3,
                 n_obstacles=(0, 3), v_max=4.0, a_max=4.0, r_agent=0.5, goal_radius=1.0, device='cpu', seed=0,
                 w_hit_agent=5.0, w_hit_obs=5.0):
        self.N, self.B, self.L, self.dt, self.T = n_agents, batch, arena, dt, max_steps
        self.k, self.m = min(k_nb, n_agents - 1), m_obs
        self.n_obs_rng, self.vmax, self.amax = n_obstacles, v_max, a_max
        self.r, self.goal_r, self.dev = r_agent, goal_radius, device
        self.w_ha, self.w_ho = w_hit_agent, w_hit_obs
        self.max_obs = max(n_obstacles[1], 1)
        self.g = torch.Generator(device='cpu').manual_seed(seed)
        self.obs_dim = 7 + 4 * self.k + 3 * self.m
        self.reset()

    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.g).to(self.dev)

    def reset(self):
        B, N, L = self.B, self.N, self.L
        c = L / 2
        R = L * (0.36 + 0.08 * self._rand(B, 1))
        ang = 2 * math.pi * torch.arange(N, device=self.dev) / N + 2 * math.pi * self._rand(B, 1)
        ang = ang + 0.15 * (self._rand(B, N) - 0.5)
        self.pos = torch.stack([c + R * torch.cos(ang), c + R * torch.sin(ang)], -1)
        self.goal = torch.stack([c - R * torch.cos(ang), c - R * torch.sin(ang)], -1)
        self.vel = torch.zeros(B, N, 2, device=self.dev)
        # obstacles: circles near the centre; inactive ones are far away
        no = torch.randint(self.n_obs_rng[0], self.n_obs_rng[1] + 1, (B,), generator=self.g).to(self.dev)
        oc = c + (self._rand(B, self.max_obs, 2) - 0.5) * L * 0.35
        orad = 1.5 + 2.0 * self._rand(B, self.max_obs)
        active = torch.arange(self.max_obs, device=self.dev)[None] < no[:, None]
        self.obs_c = torch.where(active[..., None], oc, torch.full_like(oc, 1e4))
        self.obs_r = torch.where(active, orad, torch.zeros_like(orad))
        self.t = 0
        self.arrived = torch.zeros(B, N, dtype=torch.bool, device=self.dev)
        self.collisions = torch.zeros(B, device=self.dev)
        self.prev_d = (self.goal - self.pos).norm(dim=-1)
        return self.observe()

    def observe(self):
        B, N = self.B, self.N
        to_goal = self.goal - self.pos
        dg = to_goal.norm(dim=-1, keepdim=True)
        ego = torch.cat([self.vel / self.vmax, to_goal / self.L * 2, torch.clamp(dg / 10, max=3),
                         (self.pos / self.L) * 2 - 1], -1)                               # 7
        rel = self.pos[:, None] - self.pos[:, :, None]                                   # [b,i,j] = p_j - p_i
        d = rel.norm(dim=-1) + torch.eye(N, device=self.dev) * 1e6
        dk, idx = d.topk(self.k, dim=-1, largest=False)
        rp = torch.gather(rel, 2, idx[..., None].expand(-1, -1, -1, 2))
        rv = torch.gather(self.vel[:, None].expand(-1, N, -1, -1) - self.vel[:, :, None], 2, idx[..., None].expand(-1, -1, -1, 2))
        nb = torch.cat([rp / 10, rv / self.vmax], -1)                                    # [B,N,k,4]
        orel = self.obs_c[:, None] - self.pos[:, :, None]                                # [B,N,O,2]
        od = orel.norm(dim=-1) - self.obs_r[:, None]
        odm, oidx = od.topk(min(self.m, self.max_obs), dim=-1, largest=False)
        orp = torch.gather(orel, 2, oidx[..., None].expand(-1, -1, -1, 2))
        ob = torch.cat([orp / (orel.gather(2, oidx[..., None].expand(-1, -1, -1, 2)).norm(dim=-1, keepdim=True) + 1e-6),
                        torch.clamp(odm[..., None] / 10, max=3)], -1)                     # [B,N,m',3]
        if ob.shape[2] < self.m:
            pad = torch.zeros(B, N, self.m - ob.shape[2], 3, device=self.dev); pad[..., 2] = 3
            ob = torch.cat([ob, pad], 2)
        ob[..., 2] = torch.clamp(ob[..., 2], max=3)
        far = ob[..., 2] >= 3
        ob = torch.where(far[..., None], torch.tensor([0., 0., 3.], device=self.dev), ob)
        return torch.cat([ego, nb.flatten(2), ob.flatten(2)], -1)

    def step(self, action):
        a = torch.clamp(action, -1, 1) * self.amax
        a = torch.where(self.arrived[..., None], -self.vel / self.dt * 0.5, a)          # arrived drones brake
        self.vel = self.vel + a * self.dt
        sp = self.vel.norm(dim=-1, keepdim=True)
        self.vel = self.vel * torch.clamp(self.vmax / (sp + 1e-9), max=1.0)
        self.pos = torch.clamp(self.pos + self.vel * self.dt, 0, self.L)
        self.t += 1
        d = (self.goal - self.pos).norm(dim=-1)
        # collisions
        dd = (self.pos[:, None] - self.pos[:, :, None]).norm(dim=-1) + torch.eye(self.N, device=self.dev) * 1e6
        hit_a = (dd < 2 * self.r).any(-1)
        do = (self.obs_c[:, None] - self.pos[:, :, None]).norm(dim=-1) - self.obs_r[:, None]
        hit_o = (do < self.r).any(-1)
        close = torch.clamp(1.5 - dd.min(-1).values, min=0)                              # shaping: keep a margin
        new_arr = (d < self.goal_r) & ~self.arrived
        self.arrived |= new_arr
        oclose = torch.clamp(1.0 - do.min(-1).values, min=0)                            # keep clear of obstacles
        rew = 1.0 * (self.prev_d - d) - self.w_ha * hit_a.float() - self.w_ho * hit_o.float() - 0.5 * close - 1.0 * oclose \
            + 10.0 * new_arr.float() - 0.01 * (action ** 2).sum(-1) - 0.02
        rew = torch.where(self.arrived & ~new_arr, torch.zeros_like(rew), rew)
        self.prev_d = d
        self.collisions += (hit_a | hit_o).float().sum(-1)
        done = self.t >= self.T or bool(self.arrived.all())
        info = dict(hit_agents=hit_a, hit_obs=hit_o)
        return self.observe(), rew, done, info


# ---------------------------------------------------------------- policy
class AttentionPolicy(nn.Module):
    def __init__(self, k, m, hidden=64, heads=4):
        super().__init__()
        self.k, self.m = k, m
        self.ego = nn.Sequential(nn.Linear(7, hidden), nn.Tanh())
        self.nb = nn.Sequential(nn.Linear(4, hidden), nn.Tanh())
        self.ob = nn.Sequential(nn.Linear(3, hidden), nn.Tanh())
        self.att = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.actor = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        self.critic = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.log_std = nn.Parameter(torch.full((2,), -0.5))
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)

    def features(self, obs):
        shp = obs.shape[:-1]
        o = obs.reshape(-1, obs.shape[-1])
        e = self.ego(o[:, :7])
        nb = self.nb(o[:, 7:7 + 4 * self.k].reshape(-1, self.k, 4))
        ob = self.ob(o[:, 7 + 4 * self.k:].reshape(-1, self.m, 3))
        tokens = torch.cat([nb, ob], 1)
        ctx, w = self.att(e[:, None], tokens, tokens, need_weights=True)
        h = torch.cat([e, ctx[:, 0]], -1)
        return h.reshape(*shp, -1), w.reshape(*shp, -1)

    def forward(self, obs):
        """obs [B, N, D] -> mean [B,N,2], value [B,N]"""
        h, _ = self.features(obs)
        mean = torch.tanh(self.actor(h))
        glob = h[..., :h.shape[-1] // 2].mean(-2, keepdim=True).expand(*h.shape[:-1], h.shape[-1] // 2)
        v = self.critic(torch.cat([h, glob], -1)).squeeze(-1)
        return mean, v

    def dist(self, obs):
        mean, v = self(obs)
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean)), v

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        mean, _ = self(obs)
        return mean if deterministic else torch.distributions.Normal(mean, self.log_std.exp()).sample()


# ---------------------------------------------------------------- PPO
def train_mappo(updates=150, n_agents=8, batch=32, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, epochs=4,
                minibatch=4096, ent_coef=0.003, seed=0, log_every=10, policy=None, env_kw=None, verbose=True,
                eval_every=0):
    torch.manual_seed(seed)
    env = SwarmNavEnv(n_agents=n_agents, batch=batch, seed=seed, **(env_kw or {}))
    pol = policy or AttentionPolicy(env.k, env.m)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    hist = []
    best = (-1e9, None)
    t0 = time.time()
    for u in range(1, updates + 1):
        obs = env.reset()
        O, A, LP, R, V, D = [], [], [], [], [], []
        done = False
        while not done:
            with torch.no_grad():
                dist, v = pol.dist(obs)
                a = dist.sample()
                lp = dist.log_prob(a).sum(-1)
            alive = (~env.arrived).float()
            nobs, r, done, _ = env.step(a)
            O.append(obs); A.append(a); LP.append(lp); R.append(r); V.append(v); D.append(alive)
            obs = nobs
        with torch.no_grad():
            _, last_v = pol.dist(obs)
        T = len(R)
        R, V, D = torch.stack(R), torch.stack(V), torch.stack(D)
        adv = torch.zeros_like(R); gae = 0
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else V[t + 1]
            nonterm = 0.0 if (t == T - 1 and done and env.arrived.all()) else 1.0
            delta = R[t] + gamma * nv * nonterm - V[t]
            gae = delta + gamma * lam * gae
            adv[t] = gae
        ret = adv + V
        # minibatches are whole swarms (one env at one time step) so the centralised critic
        # always sees the same mean-pooled swarm context it saw during the rollout
        N, Dd = env.N, obs.shape[-1]
        Ob = torch.stack(O).reshape(-1, N, Dd); Ac = torch.stack(A).reshape(-1, N, 2)
        Lp = torch.stack(LP).reshape(-1, N); Ad = adv.reshape(-1, N); Re = ret.reshape(-1, N)
        Mk = D.reshape(-1, N)
        Ad = (Ad - Ad[Mk > 0].mean()) / (Ad[Mk > 0].std() + 1e-8)
        rows = Ob.shape[0]
        mb = max(1, minibatch // N)
        for _ in range(epochs):
            perm = torch.randperm(rows)
            for s in range(0, rows, mb):
                i = perm[s:s + mb]
                dist, vpred = pol.dist(Ob[i])
                lp = dist.log_prob(Ac[i]).sum(-1)
                ratio = (lp - Lp[i]).exp()
                m = Mk[i]; denom = m.sum().clamp(min=1)
                pl = -(torch.min(ratio * Ad[i], torch.clamp(ratio, 1 - clip, 1 + clip) * Ad[i]) * m).sum() / denom
                vl = (((vpred - Re[i]) ** 2) * m).sum() / denom
                ent = (dist.entropy().sum(-1) * m).sum() / denom
                loss = pl + 0.5 * vl - ent_coef * ent
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(pol.parameters(), 0.5); opt.step()
        row = dict(update=u, ep_return=float(R.sum(0).mean()), success=float(env.arrived.float().mean()),
                   collisions_per_ep=float(env.collisions.mean()), steps=T, time_s=time.time() - t0)
        if eval_every and u % eval_every == 0:
            ev = evaluate(pol, n_agents=n_agents, episodes=48, seed=777, **(env_kw or {}))
            score = ev['success_rate'] - 0.05 * ev['collision_steps_per_ep']
            row['eval_score'] = score
            if score > best[0]:
                best = (score, {k: v.clone() for k, v in pol.state_dict().items()})
        hist.append(row)
        if verbose and (u % log_every == 0 or u == 1):
            print(f"upd {u:4d} | return {row['ep_return']:7.2f} | arrived {row['success']:.2f} | "
                  f"collisions/ep {row['collisions_per_ep']:.2f} | {row['time_s']:.0f}s")
    if best[1] is not None:
        pol.load_state_dict(best[1])
    return pol, hist


# ---------------------------------------------------------------- baseline + evaluation
class PotentialFieldController:
    """Goal attraction + inverse-square repulsion from neighbours and obstacles."""

    def __init__(self, k_goal=1.0, k_rep=6.0, rep_range=4.0):
        self.kg, self.kr, self.rr = k_goal, k_rep, rep_range

    def act(self, env, obs=None):
        to_g = env.goal - env.pos
        a = self.kg * to_g / (to_g.norm(dim=-1, keepdim=True) + 1e-6) * env.vmax - env.vel
        rel = env.pos[:, :, None] - env.pos[:, None]
        d = rel.norm(dim=-1) + torch.eye(env.N) * 1e6
        w = torch.where(d < self.rr, (1 / d - 1 / self.rr) / d ** 2, torch.zeros_like(d))
        a = a + self.kr * (rel * w[..., None]).sum(2)
        orel = env.pos[:, :, None] - env.obs_c[:, None]
        od = orel.norm(dim=-1) - env.obs_r[:, None]
        wo = torch.where(od < self.rr, (1 / od.clamp(min=0.1) - 1 / self.rr) / od.clamp(min=0.1) ** 2, torch.zeros_like(od))
        a = a + self.kr * (orel / (orel.norm(dim=-1, keepdim=True) + 1e-6) * wo[..., None]).sum(2)
        return torch.clamp(a / env.amax, -1, 1)


@torch.no_grad()
def evaluate(controller, n_agents=8, episodes=64, seed=123, record=False, **env_kw):
    env = SwarmNavEnv(n_agents=n_agents, batch=episodes, seed=seed, **env_kw)
    obs = env.reset(); done = False
    traj = [env.pos[0].clone()]
    first_arrival = torch.full((episodes, n_agents), float(env.T))
    hits = torch.zeros(episodes); hits_a = torch.zeros(episodes); hits_o = torch.zeros(episodes)
    min_sep = torch.full((episodes,), 1e9)
    while not done:
        if isinstance(controller, (PotentialFieldController, CBFShield)):
            a = controller.act(env, obs)
        else:
            a = controller.act(obs)
        obs, r, done, info = env.step(a)
        hits += (info['hit_agents'] | info['hit_obs']).float().sum(-1)
        hits_a += info['hit_agents'].float().sum(-1); hits_o += info['hit_obs'].float().sum(-1)
        dd = (env.pos[:, None] - env.pos[:, :, None]).norm(dim=-1) + torch.eye(n_agents) * 1e6
        min_sep = torch.minimum(min_sep, dd.flatten(1).min(-1).values)
        first_arrival = torch.where(env.arrived & (first_arrival == env.T), torch.tensor(float(env.t)), first_arrival)
        if record: traj.append(env.pos[0].clone())
    res = dict(success_rate=float(env.arrived.all(-1).float().mean()), arrived=float(env.arrived.float().mean()),
               collision_steps_per_ep=float(hits.mean()), collision_free_eps=float((hits == 0).float().mean()),
               agent_collisions_per_ep=float(hits_a.mean()), obstacle_collisions_per_ep=float(hits_o.mean()),
               mean_time_s=float(first_arrival[env.arrived].mean() * env.dt) if env.arrived.any() else float('nan'),
               min_separation=float(min_sep.min()))
    if record:
        res['traj'] = torch.stack(traj).numpy(); res['goal'] = env.goal[0].numpy()
        res['obs_c'] = env.obs_c[0].numpy(); res['obs_r'] = env.obs_r[0].numpy()
    return res


class CBFShield:
    """Safe RL: the learned policy proposes an acceleration, a control-barrier-function
    safety filter (velocity level, decentralised QP per drone) minimally corrects it so
    that drone-drone and drone-obstacle distances stay above a margin."""

    def __init__(self, policy, D_agent=1.2, margin_obs=0.8, alpha=2.0, right_bias=0.0):
        self.pol, self.D, self.mo, self.alpha, self.rb = policy, D_agent, margin_obs, alpha, right_bias

    def act(self, env, obs):
        from ..robotics.cbf import qp_project
        a = self.pol.act(obs)
        v_des = (env.vel + torch.clamp(a, -1, 1) * env.amax * env.dt).numpy()
        P = env.pos.numpy(); OC = env.obs_c.numpy(); OR = env.obs_r.numpy()
        if self.rb > 0:                      # deadlock resolution: bend to the right when crowded
            dd = np.linalg.norm(P[:, :, None] - P[:, None], axis=-1) + np.eye(env.N) * 1e9
            crowd = np.clip((3 * self.D - dd.min(-1)) / (2 * self.D), 0, 1)[..., None]
            v_des = v_des + self.rb * crowd * np.stack([v_des[..., 1], -v_des[..., 0]], -1)
        out = v_des.copy()
        for b in range(env.B):
            for i in range(env.N):
                if env.arrived[b, i]:
                    continue
                A, bb = [], []
                d = P[b, i] - P[b]
                dist2 = (d ** 2).sum(1)
                for j in np.where((dist2 < 36) & (np.arange(env.N) != i))[0]:
                    A.append(2 * d[j]); bb.append(-0.5 * self.alpha * (dist2[j] - self.D ** 2))
                for o in range(OC.shape[1]):
                    if OR[b, o] <= 0:
                        continue
                    do = P[b, i] - OC[b, o]; R = OR[b, o] + env.r + self.mo
                    if do @ do < (R + 6) ** 2:
                        A.append(2 * do); bb.append(-self.alpha * (do @ do - R ** 2))
                if A:
                    out[b, i] = qp_project(v_des[b, i], np.array(A), np.array(bb), iters=50)
        out = torch.tensor(out, dtype=torch.float32)
        sp = out.norm(dim=-1, keepdim=True)
        out = out * torch.clamp(env.vmax / (sp + 1e-9), max=1.0)
        return torch.clamp((out - env.vel) / (env.amax * env.dt), -1, 1)
