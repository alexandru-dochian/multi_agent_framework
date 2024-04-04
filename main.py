import logging
import sys
import time
from threading import Event

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

columns = [
    'stateEstimate.x',
    'stateEstimate.y',
    'stateEstimate.z',
    'stateEstimate.vx',
    'stateEstimate.vy',
    'stateEstimate.vz',
]

df: pd.DataFrame = pd.DataFrame(columns=columns)

URI = uri_helper.uri_from_env(default='radio://0/100/2M/E7E7E7E702')

DEFAULT_HEIGHT = 0.4
MAX_HEIGHT = 1.5 * DEFAULT_HEIGHT
BOX_LIMIT = 0.2

position_estimate = [0, 0, 0]

deck_attached_event = Event()

logging.basicConfig(level=logging.ERROR)

def param_deck_flow(_, value_str):
    value = int(value_str)
    print(value)
    if value:
        deck_attached_event.set()
        print('Deck is attached!')
    else:
        print('Deck is NOT attached!')

def take_off_simple(scf):
    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        time.sleep(3)
        mc.stop()

def move_linear_simple(scf):
    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        time.sleep(1)
        mc.forward(0.5)
        time.sleep(1)

def move_box_limit(scf):
    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        body_x_cmd = 1
        body_y_cmd = 1
        body_z_cmd = 0
        max_vel = 0.2
        max_z_velocity = 0.1

        total_time = 30 # seconds
        spent_time = 0
        while (spent_time < total_time):
            if position_estimate[0] > BOX_LIMIT:
                body_x_cmd = -max_vel
            elif position_estimate[0] < -BOX_LIMIT:
                body_x_cmd = max_vel
            if position_estimate[1] > BOX_LIMIT:
                body_y_cmd = -max_vel
            elif position_estimate[1] < -BOX_LIMIT:
                body_y_cmd = max_vel

            if position_estimate[2] <= DEFAULT_HEIGHT:
                body_z_cmd = max_z_velocity
            elif position_estimate[2] >= MAX_HEIGHT:
                body_z_cmd = -max_z_velocity

            mc.start_linear_motion(body_x_cmd, body_y_cmd, body_z_cmd)

            time.sleep(0.1)
            spent_time += 0.1

        df.to_csv("omg.csv", index=False)


def log_pos_callback(timestamp, data, logconf):
    global position_estimate
    global df
    # row_df = pd.DataFrame.from_dict(data)
    row_df = pd.DataFrame(data, index=[0])
    df = pd.concat([df, row_df])

    print("row_df", row_df)
    position_estimate[0] = data['stateEstimate.x']
    position_estimate[1] = data['stateEstimate.y']
    position_estimate[2] = data['stateEstimate.z']

if __name__ == '__main__':
    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:

        scf.cf.param.add_update_callback(group='deck', name='bcFlow2',
                                         cb=param_deck_flow)
        time.sleep(1)


        logconf = LogConfig(name='Position', period_in_ms=10)
        logconf.add_variable('stateEstimate.x', 'float')
        logconf.add_variable('stateEstimate.y', 'float')
        logconf.add_variable('stateEstimate.z', 'float')
        logconf.add_variable('stateEstimate.vx', 'float')
        logconf.add_variable('stateEstimate.vy', 'float')
        logconf.add_variable('stateEstimate.vz', 'float')
        
        scf.cf.log.add_config(logconf)
        logconf.data_received_cb.add_callback(log_pos_callback)

        if not deck_attached_event.wait(timeout=5):
            print('No flow deck detected!')
            sys.exit(1)

        logconf.start()
        move_box_limit(scf)
        logconf.stop()
