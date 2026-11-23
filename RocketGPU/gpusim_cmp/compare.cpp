// Side-by-side: RocketGPU (CUDA, via its RocketSim-compatible Arena) vs our RocketSim (CPU, real soccar meshes).
// Same initial state + same controls each tick; reports error percentiles at fixed tick checkpoints.
#include "rs_side.h"
#include <rocketgpu/rocketsim_compat.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace rocketgpu;
namespace rc = rocketgpu::compat;

static const int kCheck[] = {1, 10, 30, 60, 120, 240, 600};
static const char* kMesh = "D:/GigaLearnBot/collision_meshes";

struct Sides { void* rs; rc::Arena* rg; int cars; };

static rc::RotMat ToRot(const CmpBody& b) {
    rc::RotMat m; m.forward = {b.fwd[0], b.fwd[1], b.fwd[2]}; m.right = {b.right[0], b.right[1], b.right[2]};
    m.up = {b.up[0], b.up[1], b.up[2]}; return m;
}
static void FillRG(rc::PhysState& s, const CmpBody& b) {
    s.pos = {b.pos[0], b.pos[1], b.pos[2]}; s.vel = {b.vel[0], b.vel[1], b.vel[2]};
    s.angVel = {b.ang[0], b.ang[1], b.ang[2]}; s.rotMat = ToRot(b);
}
static CmpBody FromRG(const rc::PhysState& s) {
    CmpBody b{};
    float* d[] = {b.pos, b.vel, b.ang, b.fwd, b.right, b.up};
    const Vec3* v[] = {&s.pos, &s.vel, &s.angVel, &s.rotMat.forward, &s.rotMat.right, &s.rotMat.up};
    for (int i = 0; i < 6; i++) { d[i][0] = v[i]->x; d[i][1] = v[i]->y; d[i][2] = v[i]->z; }
    return b;
}
static float Dist(const float* a, const float* b) {
    return std::sqrt((a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]));
}

static Sides Make(int cars) {
    Sides s{rs_create(kMesh, cars), rc::Arena::Create(rc::GameMode::SOCCAR), cars};
    for (int i = 0; i < cars; i++) s.rg->AddCar(i % 2 ? Team::Orange : Team::Blue, OctaneConfig());
    return s;
}
static void Free(Sides& s) { rs_destroy(s.rs); delete s.rg; }

static void SetBall(Sides& s, const CmpBody& b) {
    rs_set_ball(s.rs, b);
    auto st = s.rg->ball->GetState(); FillRG(st, b); st.tickCountSinceUpdate = 0; s.rg->ball->SetState(st);
}
static void SetCar(Sides& s, int i, const CmpBody& b) {
    rs_set_car(s.rs, i, b);
    auto* car = s.rg->GetCars()[i];
    auto st = car->GetState(); FillRG(st, b); st.boost = b.boost; st.tickCountSinceUpdate = 0; car->SetState(st);
}
static void SetCtrl(Sides& s, int i, const CmpCtrl& c) {
    rs_set_ctrl(s.rs, i, c);
    CarControls k; k.throttle = c.throttle; k.steer = c.steer; k.pitch = c.pitch; k.yaw = c.yaw; k.roll = c.roll;
    k.jump = c.jump; k.boost = c.boost; k.handbrake = c.handbrake;
    s.rg->GetCars()[i]->SetControls(k);
}

static CmpBody Yawed(float x, float y, float z, float yaw, float boost) {
    CmpBody b{}; b.pos[0] = x; b.pos[1] = y; b.pos[2] = z;
    b.fwd[0] = std::cos(yaw); b.fwd[1] = std::sin(yaw);
    b.right[0] = -std::sin(yaw); b.right[1] = std::cos(yaw);
    b.up[2] = 1; b.boost = boost; return b;
}

struct Errs { std::vector<float> ball_pos[7], ball_vel[7], car_pos[7], car_vel[7]; };

static void RunEpisode(Sides& s, int ticks, std::mt19937* rng, Errs& e) {
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    int ci = 0;
    for (int t = 1; t <= ticks; t++) {
        if (rng && (t - 1) % 8 == 0)
            for (int i = 0; i < s.cars; i++) {
                CmpCtrl c{u(*rng), u(*rng), u(*rng), u(*rng), u(*rng), u(*rng) > 0.6f, u(*rng) > 0.f, u(*rng) > 0.7f};
                SetCtrl(s, i, c);
            }
        rs_step(s.rs); s.rg->Step(1);
        if (ci < 7 && t == kCheck[ci]) {
            CmpBody a = rs_get_ball(s.rs), b = FromRG(s.rg->ball->GetState());
            e.ball_pos[ci].push_back(Dist(a.pos, b.pos)); e.ball_vel[ci].push_back(Dist(a.vel, b.vel));
            for (int i = 0; i < s.cars; i++) {
                CmpBody ca = rs_get_car(s.rs, i), cb = FromRG(s.rg->GetCars()[i]->GetState());
                e.car_pos[ci].push_back(Dist(ca.pos, cb.pos)); e.car_vel[ci].push_back(Dist(ca.vel, cb.vel));
            }
            ci++;
        }
    }
}

static std::string Stat(std::vector<float> v) {
    if (v.empty()) return "-";
    std::sort(v.begin(), v.end());
    char buf[64];
    auto q = [&](float p) { return v[std::min(v.size() - 1, (size_t)(p * (v.size() - 1) + 0.5f))]; };
    if (v.size() == 1) std::snprintf(buf, sizeof buf, "%.4g", v[0]);
    else std::snprintf(buf, sizeof buf, "%.3g / %.3g / %.3g", q(0.5f), q(0.95f), v.back());
    return buf;
}
static void Report(const char* name, const Errs& e, int ticks) {
    std::printf("\n### %s\n| tick | ball pos | ball vel | car pos | car vel |\n|---|---|---|---|---|\n", name);
    for (int i = 0; i < 7 && kCheck[i] <= ticks; i++)
        std::printf("| %d | %s | %s | %s | %s |\n", kCheck[i], Stat(e.ball_pos[i]).c_str(), Stat(e.ball_vel[i]).c_str(),
                    Stat(e.car_pos[i]).c_str(), Stat(e.car_vel[i]).c_str());
    std::fflush(stdout);
}

int main(int argc, char** argv) {
    int episodes = argc > 1 ? std::atoi(argv[1]) : 32;
    std::printf("RocketGPU (CUDA) vs RocketSim (CPU), soccar meshes. Cells: single value, or median / p95 / max (UU, UU/s)\n");

    struct BallCase { const char* name; float p[3], v[3], w[3]; };
    const BallCase cases[] = {
        {"ball_floor_drop", {0, 0, 250}, {0, 0, 0}, {0, 0, 0}},
        {"ball_floor_angled", {0, 0, 250}, {500, 0, -300}, {0, 3, 0}},
        {"ball_side_wall", {3800, 0, 500}, {1500, 0, 0}, {0, 0, 0}},
        {"ball_back_wall", {0, 4800, 1000}, {0, 1500, 0}, {0, 0, 0}},
        {"ball_ceiling", {0, 0, 1600}, {0, 0, 1200}, {0, 0, 0}},
        {"ball_corner_ramp", {3450, 4250, 200}, {600, 600, -200}, {0, 0, 0}},
        {"ball_goal_post", {892.8f, 4500, 300}, {0, 1500, 0}, {0, 0, 0}},
        {"ball_crossbar", {0, 4500, 642.7f}, {0, 1500, 0}, {0, 0, 0}},
        {"ball_into_goal", {0, 4000, 200}, {0, 2000, 300}, {1, 0, 0}},
    };
    for (auto& c : cases) {
        Sides s = Make(0); Errs e;
        CmpBody b{}; for (int k = 0; k < 3; k++) { b.pos[k] = c.p[k]; b.vel[k] = c.v[k]; b.ang[k] = c.w[k]; }
        b.fwd[0] = 1; b.right[1] = 1; b.up[2] = 1;
        SetBall(s, b); RunEpisode(s, 240, nullptr, e); Report(c.name, e, 240); Free(s);
    }

    { // Random ball throws all over the arena (walls, ramps, corners, goals), no cars
        std::mt19937 rng(7); std::uniform_real_distribution<float> u(-1.f, 1.f); Errs e;
        for (int ep = 0; ep < episodes; ep++) {
            Sides s = Make(0);
            CmpBody b{}; b.pos[0] = 3000 * u(rng); b.pos[1] = 4000 * u(rng); b.pos[2] = 300 + 1200 * (u(rng) + 1) / 2;
            for (int k = 0; k < 3; k++) { b.vel[k] = 3000 * u(rng); b.ang[k] = 4 * u(rng); }
            b.fwd[0] = 1; b.right[1] = 1; b.up[2] = 1;
            SetBall(s, b); RunEpisode(s, 600, nullptr, e); Free(s);
        }
        char n[64]; std::snprintf(n, sizeof n, "ball_random_throws (%d episodes)", episodes); Report(n, e, 600);
    }

    { // 1v1 from kickoff-like spots with random controls (changed every 8 ticks)
        Errs e;
        for (int ep = 0; ep < episodes; ep++) {
            std::mt19937 rng(1000 + ep); std::uniform_real_distribution<float> u(-1.f, 1.f);
            Sides s = Make(2);
            float x = 2048 * u(rng), y = -2560 - 1500 * (u(rng) + 1) / 2, yaw = 3.14159f * u(rng);
            SetCar(s, 0, Yawed(x, y, 17.01f, yaw, 33.3f));
            SetCar(s, 1, Yawed(-x, -y, 17.01f, yaw + 3.14159265f, 33.3f));
            CmpBody b{}; b.pos[2] = 93.15f; b.fwd[0] = 1; b.right[1] = 1; b.up[2] = 1; SetBall(s, b);
            RunEpisode(s, 600, &rng, e); Free(s);
        }
        char n[64]; std::snprintf(n, sizeof n, "random_1v1 (%d episodes, cars pooled)", episodes); Report(n, e, 600);
    }
    return 0;
}
