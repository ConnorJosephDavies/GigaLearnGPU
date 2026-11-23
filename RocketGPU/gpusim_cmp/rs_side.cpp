#include "rs_side.h"
#include "RocketSim.h"
#include <vector>

using namespace RocketSim;

namespace {
struct RS { Arena* arena; std::vector<Car*> cars; };
bool g_init = false;

void ToVec(const float* f, Vec& v) { v = Vec(f[0], f[1], f[2]); }
void FromVec(const Vec& v, float* f) { f[0] = v.x; f[1] = v.y; f[2] = v.z; }

void Fill(PhysState& s, const CmpBody& b) {
    ToVec(b.pos, s.pos); ToVec(b.vel, s.vel); ToVec(b.ang, s.angVel);
    ToVec(b.fwd, s.rotMat.forward); ToVec(b.right, s.rotMat.right); ToVec(b.up, s.rotMat.up);
}
CmpBody Out(const PhysState& s) {
    CmpBody b{};
    FromVec(s.pos, b.pos); FromVec(s.vel, b.vel); FromVec(s.angVel, b.ang);
    FromVec(s.rotMat.forward, b.fwd); FromVec(s.rotMat.right, b.right); FromVec(s.rotMat.up, b.up);
    return b;
}
}

void* rs_create(const char* mesh_dir, int num_cars) {
    if (!g_init) { RocketSim::Init(mesh_dir, true); g_init = true; }
    auto* r = new RS();
    r->arena = Arena::Create(GameMode::SOCCAR);
    for (int i = 0; i < num_cars; i++)
        r->cars.push_back(r->arena->AddCar(i % 2 ? Team::ORANGE : Team::BLUE, CAR_CONFIG_OCTANE));
    return r;
}
void rs_destroy(void* h) { auto* r = (RS*)h; delete r->arena; delete r; }

void rs_set_ball(void* h, const CmpBody& b) {
    auto* r = (RS*)h; BallState s = r->arena->ball->GetState(); Fill(s, b); r->arena->ball->SetState(s);
}
void rs_set_car(void* h, int i, const CmpBody& c) {
    auto* r = (RS*)h; CarState s = r->cars[i]->GetState(); Fill(s, c); s.boost = c.boost; r->cars[i]->SetState(s);
}
void rs_set_ctrl(void* h, int i, const CmpCtrl& c) {
    auto* r = (RS*)h; CarControls& k = r->cars[i]->controls;
    k.throttle = c.throttle; k.steer = c.steer; k.pitch = c.pitch; k.yaw = c.yaw; k.roll = c.roll;
    k.jump = c.jump; k.boost = c.boost; k.handbrake = c.handbrake;
}
void rs_step(void* h) { ((RS*)h)->arena->Step(1); }
CmpBody rs_get_ball(void* h) { return Out(((RS*)h)->arena->ball->GetState()); }
CmpBody rs_get_car(void* h, int i) { auto s = ((RS*)h)->cars[i]->GetState(); CmpBody b = Out(s); b.boost = s.boost; return b; }
