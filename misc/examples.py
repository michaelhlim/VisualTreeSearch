# Under HumANavRelease/examples

import matplotlib.pyplot as plt
import numpy as np
import os
from dotmap import DotMap
from humanav.humanav_renderer import HumANavRenderer
from humanav.renderer_params import create_params as create_base_params
from humanav.renderer_params import get_surreal_texture_dir


IMAGE_SIZE = 32.
DATA_PATH = '/home/ext_drive/sampada_deglurkar/vae_stanford/'
HUMANAV_PATH = '/home/sampada_deglurkar/HumANavRelease/'


def create_params():
    p = create_base_params()

	# Set any custom parameters
    p.building_name = 'area5a' #'area3'

    p.camera_params.width = 1024
    p.camera_params.height = 1024
    p.camera_params.fov_vertical = 75.
    p.camera_params.fov_horizontal = 75.

    # The camera is assumed to be mounted on a robot at fixed height
    # and fixed pitch. See humanav/renderer_params.py for more information

    # Tilt the camera 10 degree down from the horizontal axis
    p.robot_params.camera_elevation_degree = -10

    p.camera_params.modalities = ['rgb', 'disparity']
    return p


def plot_top_view(traversible, dx_m,
                camera_pos_13, filename):
     # Compute the real_world extent (in meters) of the traversible
    extent = [0., traversible.shape[1], 0., traversible.shape[0]]
    extent = np.array(extent)*dx_m

    fig = plt.figure(figsize=(30, 10))

    # Plot the 5x5 meter occupancy grid centered around the camera
    plt.imshow(traversible, extent=extent, cmap='gray',
              vmin=-.5, vmax=1.5, origin='lower')

    #plt.xlim([camera_pos_13[0, 0]-10., camera_pos_13[0, 0]+10.])
    #plt.ylim([camera_pos_13[0, 1]-10., camera_pos_13[0, 1]+10.])

    # Plot the camera
    plt.plot(camera_pos_13[0, 0], camera_pos_13[0, 1], 'bo', markersize=10, label='Camera')
    plt.quiver(camera_pos_13[0, 0], camera_pos_13[0, 1], np.cos(camera_pos_13[0, 2]), np.sin(camera_pos_13[0, 2]))

    plt.legend()
    #plt.set_xlim([camera_pos_13[0, 0]-5., camera_pos_13[0, 0]+5.])
    #plt.set_ylim([camera_pos_13[0, 1]-5., camera_pos_13[0, 1]+5.])
    #ax.set_xticks([])
    #ax.set_yticks([])
    #ax.set_title('Topview')

    fig.savefig(filename, bbox_inches='tight', pad_inches=0)


def plot_rgb(rgb_image_1mk3, filename):
    import cv2
    #fig = plt.figure(figsize=(30, 10))

    src = rgb_image_1mk3[0].astype(np.uint8)
    src = src[:,:,::-1]   ## CV2 works in BGR space instead of RGB
    #percent by which the image is resized
    scale_percent = (IMAGE_SIZE/src.shape[0]) * 100

    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)

    # resize image
    output = cv2.resize(src, dsize)

    # Plot the RGB Image
    #plt.imshow(output)
    #plt.imshow(rgb_image_1mk3[0].astype(np.uint8))
    #ax.set_xticks([])
    #ax.set_yticks([])
    #ax.set_title('RGB')

    cv2.imwrite(filename, output)
    #cv2.imwrite('original.png', src) 
    
    #fig.savefig(filename, bbox_inches='tight', pad_inches=0)


def plot_images(rgb_image_1mk3, depth_image_1mk1, traversible, dx_m,
                camera_pos_13, human_pos_3, filename):

    # Compute the real_world extent (in meters) of the traversible
    extent = [0., traversible.shape[1], 0., traversible.shape[0]]
    extent = np.array(extent)*dx_m

    fig = plt.figure(figsize=(30, 10))

    # Plot the 5x5 meter occupancy grid centered around the camera
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(traversible, extent=extent, cmap='gray',
              vmin=-.5, vmax=1.5, origin='lower')

    # Plot the camera
    ax.plot(camera_pos_13[0, 0], camera_pos_13[0, 1], 'bo', markersize=10, label='Camera')
    ax.quiver(camera_pos_13[0, 0], camera_pos_13[0, 1], np.cos(camera_pos_13[0, 2]), np.sin(camera_pos_13[0, 2]))
    # Plot the human
    ax.plot(human_pos_3[0], human_pos_3[1], 'ro', markersize=10, label='Human')
    ax.quiver(human_pos_3[0], human_pos_3[1], np.cos(human_pos_3[2]), np.sin(human_pos_3[2]))

    ax.legend()
    ax.set_xlim([camera_pos_13[0, 0]-5., camera_pos_13[0, 0]+5.])
    ax.set_ylim([camera_pos_13[0, 1]-5., camera_pos_13[0, 1]+5.])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Topview')

    # Plot the RGB Image
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(rgb_image_1mk3[0].astype(np.uint8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('RGB')

    # Plot the Depth Image
    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(depth_image_1mk1[0, :, :, 0].astype(np.uint8), cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Depth')

    fig.savefig(filename, bbox_inches='tight', pad_inches=0)


def render_rgb_and_depth(r, camera_pos_13, dx_m, human_visible=False):
    # Convert from real world units to grid world units
    camera_grid_world_pos_12 = camera_pos_13[:, :2]/dx_m

    # Render RGB and Depth Images. The shape of the resulting
    # image is (1 (batch), m (width), k (height), c (number channels))
    rgb_image_1mk3 = r._get_rgb_image(camera_grid_world_pos_12, camera_pos_13[:, 2:3], human_visible=False)

    depth_image_1mk1, _, _ = r._get_depth_image(camera_grid_world_pos_12, camera_pos_13[:, 2:3], xy_resolution=.05, map_size=1500, pos_3=camera_pos_13[0, :3], human_visible=True)

    return rgb_image_1mk3, depth_image_1mk1


def generate_training_data_easy(num_data, path):
    i = 0
    while i < num_data:
        theta = np.random.rand() * 0.9
        if theta >= 0.2:
            camera_pos_13 = np.array([[8, 24, theta]])
            generate_one_data(camera_pos_13, path)
            i += 1


def generate_training_data_medium(num_data, path):
    i = 0
    while i < num_data: 
        theta = np.random.rand() * 2*np.pi
        camera_pos_13 = np.array([[8, 24, theta]])
        generate_one_data(camera_pos_13, path)
        i += 1


def generate_training_data_hard(num_data, path):
     # Good corridor range:
    # x: 8.0 to 9.0
    # y: 20.0 to 25.0, can extend down to 17 or 16 at least
    # any theta out of 8 cardinal directions

    thetas = [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
    i = 0
    while i < num_data:
        theta = thetas[np.random.randint(len(thetas))]
        x = np.random.rand() + 8.0
        y = np.random.rand() * 5 + 20.0
        camera_pos_13 = np.array([[x, y, theta]])
        generate_one_data(camera_pos_13, path)
        i += 1


def generate_training_data_hallway(num_data, path):
    # Good corridor range:
    # x 24 to 32.5
    # y 23 to 24.5
    # # any theta out of 8 cardinal directions

    thetas = [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
    i = 0
    while i < num_data:
        theta = thetas[np.random.randint(len(thetas))]
        x = np.random.rand() * 8.5 + 24.0
        y = np.random.rand() * 1.5 + 23.0
        camera_pos_13 = np.array([[x, y, theta]])
        generate_one_data(camera_pos_13, path)
        i += 1


def generate_training_data_analysis(num_data, path):
    theta = np.pi
    ys = np.linspace(20.0, 25.0, num_data)   
    x = 8.0
    for y in ys:
        camera_pos_13 = np.array([[x, y, theta]])
        generate_one_data(camera_pos_13, path)


def generate_one_data(camera_pos_13, path):
    p = create_params()

    r = HumANavRenderer.get_renderer(p)
    dx_cm, traversible = r.get_config()

    # Convert the grid spacing to units of meters. Should be 5cm for the S3DIS data
    dx_m = dx_cm/100.

    # State of the camera. 
    # Specified as [x (meters), y (meters), theta (radians)] coordinates
    #camera_pos_13 = np.array([[7.5, 12., -1.3]])
    #camera_pos_13 = np.array([[8, 24, 0.2]])

    rgb_image_1mk3, depth_image_1mk1 = render_rgb_and_depth(r, camera_pos_13, dx_m, human_visible=False)

    camera_pos_str = '_' + str(camera_pos_13[0][0]) + '_' + str(camera_pos_13[0][1]) + '_' + str(camera_pos_13[0][2])
    filename_topdown = 'top_view' + camera_pos_str + '.png'
    filename_rgb = 'rgb' + camera_pos_str + '.png'

    # Plot the rendered images
    path_top_down = DATA_PATH + path + '/top_downs/'
    path_rgbs = DATA_PATH + path + '/rgbs/'
    plot_top_view(traversible, dx_m, camera_pos_13, path_top_down + filename_topdown)
    plot_rgb(rgb_image_1mk3, path_rgbs + filename_rgb)


def generate_observation(camera_pos_13, path):
    p = create_params()

    r = HumANavRenderer.get_renderer(p)
    dx_cm, traversible = r.get_config()

    # Convert the grid spacing to units of meters. Should be 5cm for the S3DIS data
    dx_m = dx_cm/100.

    rgb_image_1mk3, depth_image_1mk1 = render_rgb_and_depth(r, camera_pos_13, dx_m, human_visible=False)

    camera_pos_str = '_' + str(camera_pos_13[0][0]) + '_' + str(camera_pos_13[0][1]) + '_' + str(camera_pos_13[0][2])
    filename_rgb = 'rgb' + camera_pos_str + '.png'

    # Plot the rendered images
    plot_rgb(rgb_image_1mk3, path + filename_rgb)

    return path + filename_rgb, traversible, dx_m


def example1():
    """
    Code for loading a random human into the environment
    and rendering topview, rgb, and depth images.
    """
    p = create_params()

    r = HumANavRenderer.get_renderer(p)
    dx_cm, traversible = r.get_config()

    # Convert the grid spacing to units of meters. Should be 5cm for the S3DIS data
    dx_m = dx_cm/100.

    # Set the identity seed. This is used to sample a random human identity
    # (gender, texture, body shape)
    identity_rng = np.random.RandomState(48)

    # Set the Mesh seed. This is used to sample the actual mesh to be loaded
    # which reflects the pose of the human skeleton.
    mesh_rng = np.random.RandomState(20)

    # State of the camera and the human. 
    # Specified as [x (meters), y (meters), theta (radians)] coordinates
    #camera_pos_13 = np.array([[7.5, 12., -1.3]])
    camera_pos_13 = np.array([[8, 24, 0.2]])
    #camera_pos_13 = np.array([[32.5, 24, np.pi/4]])
    human_pos_3 = np.array([8.0, 9.75, np.pi/2.])

    # Speed of the human in m/s
    human_speed = 0.7

    # Load a random human at a specified state and speed
    #r.add_human_at_position_with_speed(human_pos_3, human_speed, identity_rng, mesh_rng)

    # Get information about which mesh was loaded
    #human_mesh_info = r.human_mesh_params

    rgb_image_1mk3, depth_image_1mk1 = render_rgb_and_depth(r, camera_pos_13, dx_m, human_visible=False)

    # Remove the human from the environment
    #r.remove_human()

    camera_pos_str = '_' + str(camera_pos_13[0][0]) + '_' + str(camera_pos_13[0][1]) + '_' + str(camera_pos_13[0][2])
    filename_topdown = 'top_view' + camera_pos_str + '.png'
    filename_rgb = 'rgb' + camera_pos_str + '.png'

    # Plot the rendered images
    plot_top_view(traversible, dx_m, camera_pos_13, HUMANAV_PATH + 'top_downs/' + filename_topdown)
    plot_rgb(rgb_image_1mk3, HUMANAV_PATH + 'rgbs/' + filename_rgb)

    plot_images(rgb_image_1mk3, depth_image_1mk1, traversible, dx_m,
                camera_pos_13, human_pos_3, 'example1_15.png')


def get_known_human_identity(r):
    """
    Specify a known human identity. An identity
    is a dictionary with keys {'human_gender', 'human_texture', 'body_shape'}
    """

    # Method 1: Set a seed and sample a random identity
    identity_rng = np.random.RandomState(48)
    human_identity = r.load_random_human_identity(identity_rng)

    # Method 2: If you know which human you want to load,
    # specify the params manually (or load them from a file)
    human_identity = {'human_gender': 'male', 'human_texture': [os.path.join(get_surreal_texture_dir(), 'train/male/nongrey_male_0110.jpg')], 'body_shape': 1320}
    return human_identity

def example2():
    """
    Code for loading a specified human identity into the environment
    and rendering topview, rgb, and depth images.
    Note: Example 2 is expected to produce the same output as Example1
    """
    p = create_params()

    r = HumANavRenderer.get_renderer(p)
    dx_cm, traversible = r.get_config()

    # Convert the grid spacing to units of meters. Should be 5cm for the S3DIS data
    dx_m = dx_cm/100.

    human_identity = get_known_human_identity(r)

    # Set the Mesh seed. This is used to sample the actual mesh to be loaded
    # which reflects the pose of the human skeleton.
    mesh_rng = np.random.RandomState(20)

    # State of the camera and the human. 
    # Specified as [x (meters), y (meters), theta (radians)] coordinates
    camera_pos_13 = np.array([[7.5, 12., -1.3]])
    human_pos_3 = np.array([8.0, 9.75, np.pi/2.])

    # Speed of the human in m/s
    human_speed = 0.7

    # Load a random human at a specified state and speed
    r.add_human_with_known_identity_at_position_with_speed(human_pos_3, human_speed, mesh_rng, human_identity)

    # Get information about which mesh was loaded
    human_mesh_info = r.human_mesh_params

    rgb_image_1mk3, depth_image_1mk1 = render_rgb_and_depth(r, camera_pos_13, dx_m, human_visible=True)

    # Remove the human from the environment
    r.remove_human()

    # Plot the rendered images
    plot_images(rgb_image_1mk3, depth_image_1mk1, traversible, dx_m,
                camera_pos_13, human_pos_3, 'example2.png')


if __name__ == '__main__':
    # example1() 
    # example2() 

    generate_training_data_hallway(1000, 'training_hallway')
