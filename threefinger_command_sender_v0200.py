'''
    File name: robot-grasping
    Author: minhnc
    Date created(MM/DD/YYYY): 3/16/2018
    Last modified(MM/DD/YYYY HH:MM): 3/16/2018 3:54 PM
    Python Version: 3.5
    Other modules: [tensorflow-gpu 1.3.0]

    Copyright = Copyright (C) 2017 of NGUYEN CONG MINH
    Credits = [None] # people who reported bug fixes, made suggestions, etc. but did not actually write the code
    License = None
    Version = 0.9.0.1
    Maintainer = [None]
    Email = minhnc.edu.tw@gmail.com
    Status = Prototype # "Prototype", "Development", or "Production"
    Code Style: http://web.archive.org/web/20111010053227/http://jaynes.colorado.edu/PythonGuidelines.html#module_formatting
'''

#==============================================================================
# Imported Modules
#==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time

import socket
import json
import numpy as np
from time import sleep
from enum import Enum

#==============================================================================
# Constant Definitions
#==============================================================================
class ThreeFingerModes(Enum):
    # BASIC = 0
    # PINCH = 1
    # WIDE = 2
    # SCISSOR = 3
    BASIC, PINCH, WIDE, SCISSOR = range(4)

class ROBOTIQ():
    """"""

    def __init__(self, host=socket.gethostname(), port=2345):
        """Constructor for TX60"""
        self.host = host
        self.port = port

    def _communicate(self, command):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))
        sock.sendall(json.dumps(command).encode('utf-8'))
        response = sock.recv(1024)
        sock.close()
        return response

    # @classmethod
    def enable(self):
        command = {"cmd": "enable"}
        print('Sent: ', command)
        response = self._communicate(command)
        print('Received: ', response)
        response_status = json.loads(response.decode('utf-8'))['ret']
        return response_status, []

    # @classmethod
    def disable(self):
        command = {"cmd": "disable"}
        print('Sent: ', command)
        response = self._communicate(command)
        print('Received: ', response)
        response_status = json.loads(response.decode('utf-8'))['ret']
        return response_status, []

    def set_mode(self, mode=0):
        '''

        :param mode: mode of gripper (BASIC/PINCH/WIDE/SCISSOR : 0/1/2/3)
        :return: OK or Fail
        '''
        command = {"cmd": "mode", "mode": mode}
        print('Sent: ', command)
        response = self._communicate(command)
        print('Received: ', response)
        response_status = json.loads(response.decode('utf-8'))['ret']
        return response_status, []

    def move(self, position, speed=22, force=15):
        command = {"cmd": "move", "position": position, "speed": speed, "force": force}
        print('Sent: ', command)
        response = self._communicate(command)
        print('Received: ', response)
        response_status = json.loads(response.decode('utf-8'))['ret']
        return response_status, []

#==============================================================================
# Function Definitions
#==============================================================================

#==============================================================================
# Main function
#==============================================================================
def main(argv=None):
    print('Hello! This is ROBOTIQ Command Sender Program')

    robotiq = ROBOTIQ()
    print(robotiq.enable())
    print(robotiq.set_mode(2))
    print(robotiq.set_mode(1))
    print(robotiq.move(200))
    print(robotiq.set_mode(0))
    print(robotiq.move(250))
    print(robotiq.disable())


if __name__ == '__main__':
    main()
