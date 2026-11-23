import sys
import json
import traceback

import socket
import struct

# =======================
# Example implementation of render receiver, using RocketSimVis
# =======================

# Send to RocketSimVis
UDP_IP = "127.0.0.1"
UDP_PORT = 9273

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP

def send_data_to_rsvis(j, gamemode="soccar", extra=None):
    json_out = {}
    json_out["gamemode"] = gamemode
    json_out["ball_phys"] = j['ball']
    json_out["ball_phys"].pop('forward')
    json_out["ball_phys"].pop('right')
    json_out["ball_phys"].pop('up')
    json_out["cars"] = []
    for player in j['players']:
        json_out["cars"].append(player)
    json_out["boost_pad_states"] = j['boost_pads']
    if extra and "reward_names" in extra:
        json_out["reward_names"] = extra["reward_names"]
        json_out["player_rewards"] = extra["player_rewards"]

    sock.sendto(json.dumps(json_out).encode(), (UDP_IP, UDP_PORT))

def render_state(state_json_str):
    j = json.loads(state_json_str)
    try:
        from python_scripts import watch_stats # headless stats for tools\watch_report.py (only if GIGALEARN_WATCH_STATS set)
        watch_stats.feed(j)
    except Exception:
        traceback.print_exc()
    try:
        if 'state' in j:
            send_data_to_rsvis(j['state'], j['gamemode'], j)
        else:
            send_data_to_rsvis(j)
    except Exception as err:
        print("Exception while sending data:")
        traceback.print_exc()
