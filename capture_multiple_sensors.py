#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


"""
We provide an example code to capture data for multiple sensors in synchronous mode on [Carla](https://carla.org/) platform.
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
from matplotlib import cm
import open3d as o3d
import cv2
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
from agents.navigation.behavior_agent import BehaviorAgent

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
LABEL_COLORS = np.array([
    (255, 255, 255), # None
    (70, 70, 70),    # Building
    (100, 40, 40),   # Fences
    (55, 90, 80),    # Other
    (220, 20, 60),   # Pedestrian
    (153, 153, 153), # Pole
    (157, 234, 50),  # RoadLines
    (128, 64, 128),  # Road
    (244, 35, 232),  # Sidewalk
    (107, 142, 35),  # Vegetation
    (0, 0, 142),     # Vehicle
    (102, 102, 156), # Wall
    (220, 220, 0),   # TrafficSign
    (70, 130, 180),  # Sky
    (81, 0, 81),     # Ground
    (150, 100, 100), # Bridge
    (230, 150, 140), # RailTrack
    (180, 165, 180), # GuardRail
    (250, 170, 30),  # TrafficLight
    (110, 190, 160), # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),   # Water
    (145, 170, 100), # Terrain
]) / 255.0 # normalize each channel [0-1] since is what Open3D uses


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
        default='vehicle.lincoln.mkz_2020',
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
        default=200000,
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
        '-s', '--save',
        action='store_false',
        help='disables to save the data to the disk')
    argparser.add_argument(
        '--set_start_end',
        action='store_true',
        help='set start and destination position')

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

        vis_points = o3d.utility.Vector3dVector(points)
        vis_colors = o3d.utility.Vector3dVector(int_color)

        
        
    if 'gnss' in sensor_name:
        data = np.array([sensor_data.transform.location.x, sensor_data.transform.location.y, sensor_data.transform.location.z, \
            sensor_data.transform.rotation.pitch, sensor_data.transform.rotation.yaw, sensor_data.transform.rotation.roll, \
            sensor_data.latitude, sensor_data.longitude, sensor_data.altitude])
        if args.save:
            outputGnssPath = '../output/gnss/'
            filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
            if not os.path.exists(outputGnssPath):
                os.makedirs(outputGnssPath)
            np.savetxt(outputGnssPath+str(filename)+'.txt', data)
    if 'imu' in sensor_name:
        #print(sensor_data.accelerometer, sensor_data.gyroscope, sensor_data.compass)
        data = np.array([sensor_data.accelerometer.x, sensor_data.accelerometer.y, sensor_data.accelerometer.z, \
                         sensor_data.gyroscope.x, sensor_data.gyroscope.y, sensor_data.gyroscope.z,sensor_data.compass])
        if args.save:
            outputIMUPath = '../output/imu/'
            filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
            if not os.path.exists(outputIMUPath):
                os.makedirs(outputIMUPath)
            np.savetxt(outputIMUPath+str(filename)+'.txt', data)        
            
    if 'camera' in sensor_name:
        sensor_queue.put((sensor_data.frame, sensor_name, array))
    elif 'lidar' in sensor_name:
        sensor_queue.put((sensor_data.frame, sensor_name, vis_points, vis_colors))
    else:
        sensor_queue.put((sensor_data.frame, sensor_name))

def lidarDisplayWin():
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name='Carla Lidar',
        width=960,
        height=540,
        left=480,
        top=270)
    vis.get_render_option().background_color = [0.05, 0.05, 0.05]
    vis.get_render_option().point_size = 1
    vis.get_render_option().show_coordinate_frame = True

#    if args.show_axis:
#        add_open3d_axis(vis)

    return vis

def set_start_end_pos(world, args, mode = 'random'):
    if mode == 'random':
        all_optional_position = world.get_map().get_spawn_points()
        spawn_position = random.choice(all_optional_position)
        random.shuffle(all_optional_position)
        if all_optional_position[0] != spawn_position:
            destination = all_optional_position[0]
        else:
            destination = all_optional_position[1]
    elif mode == 'manual':
        all_optional_position = world.get_map().get_spawn_points()
        spawn_position = random.choice(all_optional_position)
        destination = random.choice(all_optional_position)
    return spawn_position, destination

def modify_vehicle_physics(actor):
    #If actor is not a vehicle, we cannot use the physics control
    try:
        physics_control = actor.get_physics_control()
        physics_control.use_sweep_wheel_collision = True
        actor.apply_physics_control(physics_control)
    except Exception:
        pass

def main():
    # We start creating the client

    args = parser()

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
        settings.fixed_delta_seconds = delta                        # hzx: set fixed time-step, one tick = 50ms, i.e. 20fps
        settings.synchronous_mode = True                            # hzx: In synchronous mode, server will wait the client to finish its current work, then update
        settings.no_rendering_mode = args.no_rendering
        world.apply_settings(settings)


        # hzx: create ego vehicle
        blueprint_library = world.get_blueprint_library()                        # hzx: create blueprint object
        vehicle_bp = blueprint_library.filter(args.filter)[0]                    # hzx: get the vehicle
        if args.set_start_end:
            vehicle_transform, vehicle_destination = set_start_end_pos(world, args, mode='manual')
        else:
            # hzx: generate the spawn position and destination randomly
            vehicle_transform, vehicle_destination = set_start_end_pos(world, args, mode='random')    
        #print('start position:', vehicle_transform)
        vehicle = world.spawn_actor(vehicle_bp, vehicle_transform)               # hzx: generate the ego vehicle
        #vehicle = world.try_spawn_actor(vehicle_bp, vehicle_transform)
        
        #modify_vehicle_physics(vehicle)
        #vehicle.set_autopilot(True)                                 # hzx: default:True, run autopilot
        #traffic_manager.ignore_lights_percentage(vehicle, 100)                   # hzx: ignore the traffic ligths   

        # hzx: add spectator for monitoring better
        spectator = world.get_spectator()
        ori_spec_tran = spectator.get_transform()
        #spec_transform = vehicle.get_transform()
        #spectator.set_transform(carla.Transform(spec_transform.location + carla.Location(z=20), carla.Rotation(pitch=-90)))

        # hzx: create agent behavior
        agent = BehaviorAgent(vehicle, behavior='normal')
        agent.set_destination(vehicle_destination.location, agent._vehicle.get_location())
        #print('start:', agent._vehicle.get_location(), 'destination:', vehicle_destination.location)

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

        # hzx: Blueprints for the IMU
        imu_bp = blueprint_library.find('sensor.other.imu')



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

        #lidar_bp.set_attribute('points_per_second', '100000')
        lidar_bp.set_attribute('upper_fov', str(args.upper_fov))
        lidar_bp.set_attribute('lower_fov', str(args.lower_fov))
        lidar_bp.set_attribute('channels', str(args.channels))
        lidar_bp.set_attribute('range', str(args.range))
        lidar_bp.set_attribute('rotation_frequency', str(1.0 / delta))
        lidar_bp.set_attribute('points_per_second', str(args.points_per_second))
        
        lidar01_transform = carla.Transform(carla.Location(x=-0.5, z=1.8) + user_offset)
        lidar01 = world.spawn_actor(lidar_bp, lidar01_transform, attach_to=vehicle)
        lidar01.listen(lambda data: sensor_callback(data, sensor_queue, "lidar01", args))
        sensor_list.append(lidar01)

        gnss_transform = carla.Transform(carla.Location(x=-0.5, z=1.8) + user_offset)
        gnss = world.spawn_actor(gnss_bp, gnss_transform, attach_to=vehicle)
        gnss.listen(lambda data: sensor_callback(data, sensor_queue, "gnss", args))
        sensor_list.append(gnss)

        imu_transform = carla.Transform(carla.Location(x=-0.5, z=1.8) + user_offset)
        imu = world.spawn_actor(imu_bp, imu_transform, attach_to=vehicle)
        imu.listen(lambda data: sensor_callback(data, sensor_queue, "imu", args))
        sensor_list.append(imu)

        print('display the lidar and camera image')
        # hzx: lidar display window
        point_list = o3d.geometry.PointCloud()
        vis = lidarDisplayWin()
        # hzx: image display window
        cv2.namedWindow("front_cam", 0)
        #cv2.resizeWindow('front_cam', 400, 300)

        frame = 0
        # Main loop
        while True:
            # Tick the server
            #agent._update_information()

            world.tick()
            w_frame = world.get_snapshot().frame
            #print("\nWorld's frame: %d" % w_frame)

            #print('length waypoints queue:', len(agent._local_planner._waypoints_queue))
            #if len(agent._local_planner._waypoints_queue) < 1:


            #control = agent.run_step()
            #vehicle.apply_control(control)
            #if agent.done():
            #    print('Arrive at the target point!')
            #    exit()

            speed_limit = vehicle.get_speed_limit()
            agent.get_local_planner().set_speed(speed_limit)

            control = agent.run_step(debug=True)
            #control.manual_gear_shift = False
            vehicle.apply_control(control)

            # hzx: set the sectator to follow the ego vehicle
            spec_transform = vehicle.get_transform()
            spectator.set_transform(carla.Transform(spec_transform.location + carla.Location(z=20), carla.Rotation(pitch=-90)))
            print(vehicle_destination.location, spec_transform.location)

            # Now, we wait to the sensors data to be received.
            # As the queue is blocking, we will wait in the queue.get() methods
            # until all the information is processed and we continue with the next frame.
            # We include a timeout of 1.0 s (in the get method) and if some information is
            # not received in this time we continue.
            try:
                for _ in range(len(sensor_list)):
                    s_frame = sensor_queue.get(True, 1.0)                           # hzx: neccessary for synchronous mode
                    #print("    Frame: %d   Sensor: %s" % (s_frame[0], s_frame[1]))
                    if 'camera' in s_frame[1]:
                        im_dis = s_frame[2][:, :, ::-1]
                        cv2.imshow("front_cam",im_dis)
                        cv2.waitKey(1)

                    if 'lidar' in s_frame[1]:
                        point_list.points = s_frame[2]
                        point_list.colors = s_frame[3]
                        if frame == 0:
                            vis.add_geometry(point_list)
                        vis.update_geometry(point_list)

                        vis.poll_events()
                        vis.update_renderer()
                        # # This can fix Open3D jittering issues:
                        time.sleep(0.001)
                frame += 1



            except Empty:
                print("Some of the sensor information is missed")
            # hzx: for other exceptions, break the loop and release the resources.
            except:            
                print('other exception')                     
                break

    finally:
        world.apply_settings(original_settings)     # hzx: back to original setting, otherwise, the world will crash as it can't find the synchronous client
        vehicle.destroy()
        spectator.set_transform(ori_spec_tran)
        for sensor in sensor_list:
            sensor.destroy()
        vis.destroy_window()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')


