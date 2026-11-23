#pragma once
// Plain-data bridge to our RocketSim (same source as GigaLearn's copy) so the comparison harness
// never mixes RocketSim/Bullet headers with RocketGPU headers in one translation unit.
struct CmpBody {
    float pos[3], vel[3], ang[3];
    float fwd[3], right[3], up[3];
    float boost;
};
struct CmpCtrl { float throttle, steer, pitch, yaw, roll; bool jump, boost, handbrake; };

void* rs_create(const char* mesh_dir, int num_cars);  // car i: team = i % 2
void rs_destroy(void* h);
void rs_set_ball(void* h, const CmpBody& b);
void rs_set_car(void* h, int i, const CmpBody& c);
void rs_set_ctrl(void* h, int i, const CmpCtrl& c);
void rs_step(void* h);
CmpBody rs_get_ball(void* h);
CmpBody rs_get_car(void* h, int i);
