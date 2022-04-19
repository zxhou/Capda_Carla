#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


"""
Sensor synchronization example for CARLA

The communication model for the syncronous mode in CARLA sends the snapshot
of the world and the sensors streams in parallel.
We provide this script as an example of how to syncrononize the sensor
data gathering in the client.
To to this, we create a queue that is being filled by every sensor when the
client receives its data and the main loop is blocked until all the sensors
have received its data.
This suppose that all the sensors gather information at every tick. It this is
not the case, the clients needs to take in account at each frame how many
sensors are going to tick at each frame.
"""

import glob
import os
import sys
import argparse
import random
import time
from datetime import datetime
import numpy as np
from PIL import Image
from queue import Queue
from queue import Empty

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


def parser():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host CARLA Simulator (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port of CARLA Simulator (default: 2000)')
    argparser.add_argument(
        '--no-rendering',
        action='store_true',
        help='use the no-rendering mode which will provide some extra'
        ' performance but you will lose the articulated objects in the'
        ' lidar, such as pedestrians')
    argparser.add_argument(
        '--semantic',
        action='store_true',
        help='use the semantic lidar instead, which provides ground truth'
        ' information')
    argparser.add_argument(
        '--no-noise',
        action='store_true',
        help='remove the drop off and noise from the normal (non-semantic) lidar')
    argparser.add_argument(
        '--no-autopilot',
        action='store_false',
        help='disables the autopilot so the vehicle will remain stopped')
    argparser.add_argument(
        '--show-axis',
        action='store_true',
        help='show the cartesian coordinates axis')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='model3',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--upper-fov',
        default=15.0,
        type=float,
        help='lidar\'s upper field of view in degrees (default: 15.0)')
    argparser.add_argument(
        '--lower-fov',
        default=-25.0,
        type=float,
        help='lidar\'s lower field of view in degrees (default: -25.0)')
    argparser.add_argument(
        '--channels',
        default=64.0,
        type=float,
        help='lidar\'s channel count (default: 64)')
    argparser.add_argument(
        '--range',
        default=100.0,
        type=float,
        help='lidar\'s maximum range in meters (default: 100.0)')
    argparser.add_argument(
        '--points-per-second',
        default=500000,
        type=int,
        help='lidar\'s points per second (default: 500000)')
    argparser.add_argument(
        '-x',
        default=0.0,
        type=float,
        help='offset in the sensor position in the X-axis in meters (default: 0.0)')
    argparser.add_argument(
        '-y',
        default=0.0,
        type=float,
        help='offset in the sensor position in the Y-axis in meters (default: 0.0)')
    argparser.add_argument(
        '-z',
        default=0.0,
        type=float,
        help='offset in the sensor position in the Z-axis in meters (default: 0.0)')
    argparser.add_argument(
        '--im-width',
        default=1920,
        type=int,
        help='image width (default: 1920)'
    )
    argparser.add_argument(
        '--im-height',
        default=1080,
        type=int,
        help='image height (default: 1080)'
    )
    argparser.add_argument(
        '-s', '--save',
        action='store_false',
        help='disables to save the data to the disk')
        
    return argparser.parse_args()


# Sensor callback.
# This is where you receive the sensor data and
# process it as you liked and the important part is that,
# at the end, it should include an element into the sensor queue.
def sensor_callback(sensor_data, sensor_queue, sensor_name, args):
    # Do stuff with the sensor_data data like save it to disk
    # Then you just need to add to the queue

    if 'camera' in sensor_name:
         array = np.frombuffer(sensor_data.raw_data, dtype=np.dtype("uint8"))
         array = np.reshape(array, (600, 800, 4))
         array = array[:, :, :3]
         array = array[:, :, ::-1]
         # array = pygame.surfarray.make_surface(array.swapaxes(0, 1))
         im = Image.fromarray(array)
         if args.save:
            outputImgPath="../output/img/"
            filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
            if not os.path.exists(outputImgPath):
                os.makedirs(outputImgPath)
            im.save(outputImgPath+str(filename)+'.jpg')
        # sensor_data.save_to_disk(os.path.join('../outputs/output_synchronized', '%06d.png' % sensor_data.frame))

    if 'lidar' in sensor_name:
        """Prepares a point cloud with intensity
        colors ready to be consumed by Open3D"""
        data = np.copy(np.frombuffer(sensor_data.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))
        if args.save:
            outputLidarPath="../output/lidar/"
            filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
            if not os.path.exists(outputLidarPath):
                os.makedirs(outputLidarPath)
            np.savetxt(outputLidarPath+str(filename)+'.txt', data)

        '''
        # Isolate the intensity and compute a color for it
        intensity = data[:, -1]
        intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
        int_color = np.c_[
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

        # Isolate the 3D data
        points = data[:, :-1]


        
        # We're negating the y to correclty visualize a world that matches
        # what we see in Unreal since Open3D uses a right-handed coordinate system
        points[:, :1] = -points[:, :1]

        # # An example of converting points from sensor to vehicle space if we had
        # # a carla.Transform variable named "tran":
        # points = np.append(points, np.ones((points.shape[0], 1)), axis=1)
        # points = np.dot(tran.get_matrix(), points.T).T
        # points = points[:, :-1]

        point_list.points = o3d.utility.Vector3dVector(points)
        point_list.colors = o3d.utility.Vector3dVector(int_color)
        '''
    if 'gnss' in sensor_name:
        data = np.array([sensor_data.latitude, sensor_data.longitude, sensor_data.altitude])
        if args.save:
            outputGnssPath = '../output/gnss/'
            filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
            if not os.path.exists(outputGnssPath):
                os.makedirs(outputGnssPath)
            np.savetxt(outputGnssPath+str(filename)+'.txt', data)

    sensor_queue.put((sensor_data.frame, sensor_name))



def main():
    # We start creating the client

    args = parser()
    #args.width, args.height = [int(x) for x in args.res.split('x')]
    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)
    world = client.get_world()

    try:
        # We need to save the settings to be able to recover them at the end
        # of the script to leave the server in the same state that we found it.
        original_settings = world.get_settings()
        settings = world.get_settings()
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)                  # hzx: In synchronous mode, autopilot depends on the traffic manager

        # We set CARLA syncronous mode
        delta = 0.05
        settings.fixed_delta_seconds = delta                        # hzx: set fixed time-step, one tick = 5ms, i.e. 20fps
        settings.synchronous_mode = True                            # hzx: In synchronous mode, server will wait the client to finish its current work, then update
        settings.no_rendering_mode = args.no_rendering
        world.apply_settings(settings)


        # hzx: create ego vehicle
        blueprint_library = world.get_blueprint_library()                        # hzx: create blueprint object
        vehicle_bp = blueprint_library.filter(args.filter)[0]                    # hzx: get the vehicle(model3)
        vehicle_transform = random.choice(world.get_map().get_spawn_points())    # hzx: generate the spawn position randomly
        vehicle = world.spawn_actor(vehicle_bp, vehicle_transform)               # hzx: generate the ego vehicle
        vehicle.set_autopilot(args.no_autopilot)                                 # hzx: default:True, run autopilot
        traffic_manager.ignore_lights_percentage(vehicle, 100)                   # hzx: ignore the traffic ligths   

        # hzx: add spectator for monitoring better
        spectator = world.get_spectator()
        #spec_transform = vehicle.get_transform()
        #spectator.set_transform(carla.Transform(spec_transform.location + carla.Location(z=20), carla.Rotation(pitch=-90)))

        # We create the sensor queue in which we keep track of the information
        # already received. This structure is thread safe and can be
        # accessed by all the sensors callback concurrently without problem.
        sensor_queue = Queue()

        # Bluepints for the sensors
        blueprint_library = world.get_blueprint_library()
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        #camera_bp.set_attribute('image_size_x', str(1920))                      # hzx: image width
        #camera_bp.set_attribute('image_size_y', str(1080))                      # hzx: image height
        #camera_bp.set_attribute('fov', '90')                                    # hzx: image field of view
        #camera_bp.set_attribute('sensor_tick', str(0.04))                       # hzx: simulation seconds between sensor captures (ticks)

        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')

        # hzx: Blueprints for the gnss
        gnss_bp = blueprint_library.find('sensor.other.gnss')



        user_offset = carla.Location(args.x, args.y, args.z)

        # We create all the sensors and keep them in a list for convenience.
        sensor_list = []

        # camera relative position related to the vehicle
        #camera_transform = carla.Transform(carla.Location(0, 0, 2))
        camera_transform = carla.Transform(carla.Location(x=-0.5, z=1.8) + user_offset)

        cam01 = world.spawn_actor(cam_bp, camera_transform, attach_to=vehicle)
        # set the callback function
        cam01.listen(lambda data: sensor_callback(data, sensor_queue, "camera01", args))
        sensor_list.append(cam01)

        lidar_bp.set_attribute('points_per_second', '100000')
        lidar01_transform = carla.Transform(carla.Location(x=-0.5, z=1.8) + user_offset)
        lidar01 = world.spawn_actor(lidar_bp, lidar01_transform, attach_to=vehicle)
        lidar01.listen(lambda data: sensor_callback(data, sensor_queue, "lidar01", args))
        sensor_list.append(lidar01)

        gnss_transform = carla.Transform(carla.Location(x=-0.5, z=1.8) + user_offset)
        gnss = world.spawn_actor(gnss_bp, gnss_transform, attach_to=vehicle)
        gnss.listen(lambda data: sensor_callback(data, sensor_queue, "gnss", args))
        sensor_list.append(gnss)


        # Main loop
        while True:
            # Tick the server
            world.tick()
            w_frame = world.get_snapshot().frame
            print("\nWorld's frame: %d" % w_frame)

            # hzx: set the sectator to follow the ego vehicle
            spec_transform = vehicle.get_transform()
            spectator.set_transform(carla.Transform(spec_transform.location + carla.Location(z=20), carla.Rotation(pitch=-90)))

            # Now, we wait to the sensors data to be received.
            # As the queue is blocking, we will wait in the queue.get() methods
            # until all the information is processed and we continue with the next frame.
            # We include a timeout of 1.0 s (in the get method) and if some information is
            # not received in this time we continue.
            try:
                for _ in range(len(sensor_list)):
                    s_frame = sensor_queue.get(True, 1.0)                           # hzx: neccessary for synchronous mode
                    print("    Frame: %d   Sensor: %s" % (s_frame[0], s_frame[1]))

            except Empty:
                print("    Some of the sensor information is missed")

    finally:
        world.apply_settings(original_settings)     # hzx: back to original setting, otherwise, the world will crash as it can't find the synchronous client
        vehicle.destroy()
        for sensor in sensor_list:
            sensor.destroy()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')


