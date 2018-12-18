
from carla.agent.agent import Agent
from carla.client import VehicleControl
from random import  uniform
from carla import image_converter


import sys
sys.path.insert(0, "imitation-learning-master/agents/imitation/")
sys.path.insert(0, "models/learning_to_see/")

from imitation_learning import ImitationLearning
from enhance_dark import enhance
from model import SeeInDark

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data

import numpy as np
import cv2

# device = "cuda"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


########################## MODEL DEFINITIONS ########################################################
class base_model(nn.Module):
    def __init__(self):
        super(base_model, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride = 2, bias = False),
            nn.ReLU(),

            nn.Conv2d(24, 36, 5, stride = 2, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(36),

            nn.Conv2d(36, 48, 5, stride = 2, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(48),

            nn.Conv2d(48, 64, 5, stride = 2, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 5, stride = 2, bias = False),
            nn.ReLU(),
            nn.Dropout(p=0.35)
        )

        self.linear_net = nn.Sequential(
            nn.Linear(64 * 22 * 15, 100, bias = False),
            nn.ReLU(),
            nn.Linear(100, 50, bias = False),
            nn.ReLU(),
            nn.Linear(50, 10, bias = False),
            nn.ReLU(),
            nn.Linear(10, 1, bias = False)
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(-1, 64 * 15 * 22)
        x = self.linear_net(x)
        return x

class base_latest(nn.Module):
    def __init__(self):
        super(base_latest, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride = 2, bias = False),
            nn.BatchNorm2d(32),
            # nn.Dropout2d(0.35),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, stride = 1, bias = False),
            nn.BatchNorm2d(32),
            # nn.Dropout2d(0.35),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride = 2, bias = False),
            nn.BatchNorm2d(64),
            # nn.Dropout2d(0.35),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, stride = 1, bias = False),
            nn.BatchNorm2d(64),
            # nn.Dropout2d(0.35),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride = 2, bias = False),
            nn.BatchNorm2d(128),
            # nn.Dropout2d(0.35),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, stride = 1, bias = False),
            nn.BatchNorm2d(128),
            # nn.Dropout2d(0.35),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, stride = 1, bias = False),
            nn.BatchNorm2d(256),
            # nn.Dropout2d(0.35),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, stride = 1, bias = False),
            nn.BatchNorm2d(256),
            # nn.Dropout2d(0.35),
            nn.ReLU()
        )

        self.speed_net = nn.Sequential(
            nn.Linear(8192, 1, bias = False),
            nn.ReLU(),
            nn.Linear(100, 50, bias = False),
            nn.ReLU(),
            nn.Linear(50, 10, bias = False),
            nn.ReLU(),
            nn.Linear(10, 1, bias = False)
        )


        self.acceleration_net = nn.Sequential(
            nn.Linear(8192, 100, bias = False),
            nn.ReLU(),
            nn.Linear(100, 50, bias = False),
            nn.ReLU(),
            nn.Linear(50, 10, bias = False),
            nn.ReLU(),
            nn.Linear(10, 1, bias = False)
        )


        self.steer_net = nn.Sequential(
            nn.Linear(8192, 1, bias = False),
            # nn.ReLU(),
            # nn.Linear(100, 50, bias = False),
            # nn.ReLU(),
            # nn.Linear(50, 10, bias = False),
            # nn.ReLU(),
            # nn.Linear(10, 1, bias = False)
        )

        self.break_net = nn.Sequential(
            nn.Linear(8192, 100, bias = False),
            nn.ReLU(),
            nn.Linear(100, 50, bias = False),
            nn.ReLU(),
            nn.Linear(50, 10, bias = False),
            nn.ReLU(),
            nn.Linear(10, 1, bias = False)
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(-1, 8192)
        out = self.steer_net(x)
        return out

class immitation_model(nn.Module):
    def __init__(self):
        super(immitation_model, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride = 2, bias = False),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.35),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, stride = 1, bias = False),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.35),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride = 2, bias = False),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.35),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, stride = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.35),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride = 2, bias = False),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.35),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, stride = 1, bias = False),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.35),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, stride = 1, bias = False),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.35),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, stride = 1, bias = False),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.35),
            nn.ReLU()
        )

        self.speed_net = nn.Sequential(
            nn.Linear(8192, 100, bias = False),
            nn.ReLU(),
            nn.Linear(100, 50, bias = False),
            nn.ReLU(),
            nn.Linear(50, 10, bias = False),
            nn.ReLU(),
            nn.Linear(10, 1, bias = False)
        )


        self.acceleration_net = nn.Sequential(
            nn.Linear(8192, 100, bias = False),
            nn.ReLU(),
            nn.Linear(100, 50, bias = False),
            nn.ReLU(),
            nn.Linear(50, 10, bias = False),
            nn.ReLU(),
            nn.Linear(10, 1, bias = False)
        )


        self.steer_net = nn.Sequential(
            nn.Linear(8192, 100, bias = False),
            nn.ReLU(),
            nn.Linear(100, 50, bias = False),
            nn.ReLU(),
            nn.Linear(50, 10, bias = False),
            nn.ReLU(),
            nn.Linear(10, 1, bias = False)
        )

        self.brake_net = nn.Sequential(
            nn.Linear(8192, 100, bias = False),
            nn.ReLU(),
            nn.Linear(100, 50, bias = False),
            nn.ReLU(),
            nn.Linear(50, 10, bias = False),
            nn.ReLU(),
            nn.Linear(10, 1, bias = False)
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(-1, 8192)
        out_steer = self.steer_net(x)
        out_speed = self.speed_net(x)
        out_brake = self.brake_net(x)
        out_acceleration = self.acceleration_net(x)
        return out_steer, out_speed, out_brake, out_acceleration


################### ARTIFICIAL NIGHT ###########################################################
def adjust_gamma(image, gamma=0.4):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

########################## AGENT DEFINITIONS ########################################################
# Our first model
class CNNBaseAgent(Agent):
    """
    Base model agent - only using cnn for steering angle prediction
    """
    def __init__(self):
        # self.transform = transforms.ToTensor()
        self.model = base_model()
        # self.model = base_latest()
        self.model = self.model.to(device)
        self.mode = "Dark" # change to None for not enhancing the images through LTSID

        # Initialize LTSID model
        if self.mode and self.mode == "Dark":
            self.ltsid = SeeInDark()
            self.ltsid.load_state_dict(torch.load("/home/mihir/Downloads/CARLA_0.8.2/PythonClient/models/learning_to_see/checkpoint_sony_e3300.pth" ,map_location={'cuda:1':'cuda:0'}))
            self.ltsid = self.ltsid.to(device)

        # Old model
        # self.model.load_state_dict(torch.load('./models/cnn_base_1.model'))
        
        # New model
        self.model.load_state_dict(torch.load('./models/cnn_base_2.model'))

        # Latest model
        # self.model.load_state_dict(torch.load('./models/cnn_base_latest.model'))

    def run_step(self, measurements, sensor_data, directions, target):
        
        # Set throttle
        control = VehicleControl()
        control.throttle = 0.5
        
        # Process image from carla and pass through model
        
        img = image_converter.to_rgb_array(sensor_data['CameraRGB'])
        temp = cv2.cvtColor(img, cv2.COLOR_RGB2BGR);

        # img = adjust_gamma(img)
        cv2.imwrite('original.png',temp)
        # If Dark mode enabled, pass through learning to see in the dark model
        if self.mode and self.mode == "Dark":
            # print("Before ltsid: ",img.shape)
            # Resize to 512x512
            img = cv2.resize(img, (512, 512))
            # print("Resized: ",img.shape)
            # Pass to LTSID
            img = enhance(img, self.ltsid)
            # print("After LTSID:", img.shape)
            # Go back to 600x800x3
            img = cv2.resize(img, (600, 800))
            print("After LTSID and after reshape:", img.shape)

        img = torch.from_numpy(np.flip(img.transpose((2, 0, 1)),axis=0).copy())
        img = img.unsqueeze(0).type(torch.cuda.FloatTensor)
        
        # Set steer angle
        steer = self.model(img)
        control.steer = steer

        print("steer:{:.5f}".format(steer.item()))
        
        return control



# All Controls model
class CNNAllControlAgent(Agent):
    """
    cnn for all controls
    """
    def __init__(self):
        # self.transform = transforms.ToTensor()
        # self.model = immitation_model()
        self.model = base_latest()
        self.model = self.model.to(device)
        self.mode = None#"Dark" # change to None for not enhancing the images through LTSID

        # Initialize LTSID model
        if self.mode and self.mode == "Dark":
            self.ltsid = SeeInDark()
            self.ltsid.load_state_dict(torch.load("/home/mihir/Downloads/CARLA_0.8.2/PythonClient/models/learning_to_see/checkpoint_sony_e3300.pth" ,map_location={'cuda:1':'cuda:0'}))
            self.ltsid = self.ltsid.to(device)

        # self.model.load_state_dict(torch.load('./models/cnn_base/cnn_base.model'))
        self.model.load_state_dict(torch.load('./models/cnn_base_latest.model'))

    def run_step(self, measurements, sensor_data, directions, target):
        
        # Set throttle
        control = VehicleControl()
        control.throttle = 0.5
        
        # Process image from carla and pass through model
        img = image_converter.to_rgb_array(sensor_data['CameraRGB'])
        
        # If Dark mode enabled, pass through learning to see in the dark model
        if self.mode and self.mode == "Dark":

            img = adjust_gamma(img)

            # print("Before ltsid: ",img.shape)
            # Resize to 512x512
            img = cv2.resize(img, (512, 512))
            # print("Resized: ",img.shape)
            # Pass to LTSID
            img = enhance(img, self.ltsid)
            # print("After LTSID:", img.shape)
            # Go back to 88x200x3
            img = cv2.resize(img, (88, 200))
            print("After LTSID and after reshape:", img.shape)

        img = cv2.resize(img, (88, 200))
        img = torch.from_numpy(np.flip(img.transpose((2, 0, 1)),axis=0).copy())
        img = img.unsqueeze(0).type(torch.cuda.FloatTensor)
        
        # Set steer angle
        # steer, speed, brake, acceleration = self.model(img)
        steer = self.model(img)
        control.steer = steer
        # control.speed = speed
        # control.brake = brake
        control.throttle = 0.5

        # print("steer:{:.5f}, speed:{:.5f}, brake:{:.5f}, throttle:{:.5f}".format(steer.item(), speed.item(),brake.item(), acceleration.item()))
        
        return control

# Tensorflow Agent
class TFAgent(Agent):

    def __init__(self):
        self.obj = ImitationLearning()
        # test_print()


    def run_step(self, measurements, sensor_data, directions, target):
        
        # Set throttle
        control = VehicleControl()
        control.throttle = 0.5
        
        # Process image from carla and pass through model
        # img = image_converter.to_rgb_array(sensor_data['CameraRGB'])
        control = self.obj._compute_action(sensor_data['CameraRGB'].data,
                                       measurements.player_measurements.forward_speed, directions)
        
        return control
