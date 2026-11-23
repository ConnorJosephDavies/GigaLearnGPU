"""Headless gameplay + reward statistics for watch mode (used by tools\\watch_report.py).

render_receiver.py feeds every rendered step in here when the env var GIGALEARN_WATCH_STATS names an output JSON file.
Collected per team: weighted reward totals (what actually pays), touches / air touches / high-ball touches, boost
habits, time airborne, goals, kickoff first-touch wins and shot speed. Written to the JSON file every ~2s.
"""
import json, math, os, time

KICKOFF_EPS = 1.0
GOAL_Y = 5124 + 92  # ball centre past the goal line


class WatchStats:
    def __init__(self, path):
        self.path = path
        self.started = time.time()
        self.steps = 0
        self.reward_names = []
        self.teams = {0: self._team(), 1: self._team()}
        self.kickoffs = 0
        self.kickoff_pending = False
        self.goal_latched = False
        self.last_write = 0

    @staticmethod
    def _team():
        return {"rewards": {}, "touches": 0, "air_touches": 0, "high_ball_touches": 0, "ground_touches": 0,
                "touch_ball_speed_sum": 0.0, "goals": 0, "kickoff_first_touch": 0,
                "boost_sum": 0.0, "low_boost_steps": 0, "full_boost_steps": 0, "airborne_steps": 0,
                "high_ball_chances": 0, "high_ball_jumps": 0}

    def update(self, j):
        state = j.get("state", j)
        players = state["players"]
        ball = state["ball"]
        bpos, bvel = ball["pos"], ball["vel"]
        bspeed = math.sqrt(sum(v * v for v in bvel))
        self.steps += 1

        # Kickoff: ball parked at centre -> the next toucher wins the kickoff
        if abs(bpos[0]) < KICKOFF_EPS and abs(bpos[1]) < KICKOFF_EPS and bspeed < 1:
            if not self.kickoff_pending:
                self.kickoffs += 1
            self.kickoff_pending = True
            self.goal_latched = False

        # Goal (ball centre behind a goal line, latched until the next kickoff)
        if not self.goal_latched and abs(bpos[1]) > GOAL_Y and abs(bpos[0]) < 893:
            scorer = 0 if bpos[1] > 0 else 1  # blue scores into the orange (+y) net
            self.teams[scorer]["goals"] += 1
            self.goal_latched = True

        for p in players:
            t = self.teams[p["team_num"]]
            boost = p.get("boost", p.get("boost_amount", 0) * 100)
            t["boost_sum"] += boost
            t["low_boost_steps"] += boost < 20
            t["full_boost_steps"] += boost > 90
            t["airborne_steps"] += not p["on_ground"]
            # High-ball chance: ball 250-700 up, within 1000uu horizontally of this car
            ppos = p["phys"]["pos"]
            flat = math.hypot(bpos[0] - ppos[0], bpos[1] - ppos[1])
            if 250 < bpos[2] < 700 and flat < 1000:
                t["high_ball_chances"] += 1
                t["high_ball_jumps"] += not p["on_ground"]
            if p.get("ball_touched"):
                t["touches"] += 1
                t["touch_ball_speed_sum"] += bspeed
                if not p["on_ground"]:
                    t["air_touches"] += 1
                else:
                    t["ground_touches"] += 1
                if bpos[2] > 250:
                    t["high_ball_touches"] += 1
                if self.kickoff_pending:
                    t["kickoff_first_touch"] += 1
                    self.kickoff_pending = False

        names = j.get("reward_names")
        rewards = j.get("player_rewards")
        if names and rewards:
            self.reward_names = names
            for i, p in enumerate(players):
                if i >= len(rewards):
                    break
                tr = self.teams[p["team_num"]]["rewards"]
                for name, val in zip(names, rewards[i]):
                    tr[name] = tr.get(name, 0.0) + val

        now = time.time()
        if now - self.last_write > 2:
            self.last_write = now
            self.write()

    def summary(self):
        out = {"seconds_watched": round(time.time() - self.started, 1), "steps": self.steps,
               "kickoffs": self.kickoffs, "teams": {}}
        n = max(self.steps, 1)
        for tid, t in self.teams.items():
            touches = max(t["touches"], 1)
            rew = t["rewards"]
            total_abs = sum(abs(v) for v in rew.values()) or 1
            out["teams"]["blue" if tid == 0 else "orange"] = {
                "goals": t["goals"],
                "kickoff_first_touch": t["kickoff_first_touch"],
                "touches": t["touches"],
                "air_touch_share": round(t["air_touches"] / touches, 3),
                "high_ball_touches": t["high_ball_touches"],
                "avg_ball_speed_at_touch_kph": round(t["touch_ball_speed_sum"] / touches * 0.036, 1),
                "avg_boost": round(t["boost_sum"] / n, 1),
                "low_boost_time_share": round(t["low_boost_steps"] / n, 3),
                "full_boost_time_share": round(t["full_boost_steps"] / n, 3),
                "airborne_time_share": round(t["airborne_steps"] / n, 3),
                "jumped_at_high_ball_share": round(t["high_ball_jumps"] / max(t["high_ball_chances"], 1), 3),
                # Weighted reward totals, biggest first, with each one's share of all reward paid
                "rewards": {k: {"total": round(v, 1), "share": round(abs(v) / total_abs, 3)}
                            for k, v in sorted(rew.items(), key=lambda kv: -abs(kv[1]))},
            }
        return out

    def write(self):
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.summary(), f, indent=1)
            os.replace(tmp, self.path)
        except OSError:
            pass


_stats = None


def feed(j):
    global _stats
    path = os.environ.get("GIGALEARN_WATCH_STATS")
    if not path:
        return
    if _stats is None:
        _stats = WatchStats(path)
    _stats.update(j)
