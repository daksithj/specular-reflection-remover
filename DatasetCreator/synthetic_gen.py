# In order to run this run the following command in the terminal
# blender --background --python synthetic_gen.py -- <dataset_name> <resolution_percentage> <num_samples> <num_data>
# Example: blender --background --python synthetic_gen.py -- test 50 120 50

import bpy
import sys
from mathutils import Matrix
import os
import pickle as pkl
import numpy as np


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils
import random

file_path = os.path.abspath(__file__)
main_directory = os.path.dirname(file_path)

dataset_name = sys.argv[sys.argv.index('--') + 1]

main_directory = main_directory.replace(os.sep, '/')

# The path relative to system must be entered here. "/Output" means C:/Output/ in a Windows system
RENDER_DIR = main_directory + "/Output/" + dataset_name

SPECULAR_DIR = RENDER_DIR + "/Specular/"

DIFFUSE_DIR = RENDER_DIR + "/Diffuse/"

MATRIX_DIR = RENDER_DIR + "/Matrix/"

LOCATIONS_DIR = RENDER_DIR + "/Locations/"


def enable_gpus(device_type, use_cpus=False):
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cuda_devices, opencl_devices = cycles_preferences.get_devices()

    if device_type == "CUDA":
        devices = cuda_devices
    elif device_type == "OPENCL":
        devices = opencl_devices
    else:
        raise RuntimeError("Unsupported device type")

    activated_gpus = []

    for device in devices:
        if device.type == "CPU":
            device.use = use_cpus
        else:
            device.use = True
            activated_gpus.append(device.name)

    cycles_preferences.compute_device_type = device_type
    bpy.context.scene.cycles.device = "GPU"

    return activated_gpus


def set_principled_node_as_diffuse(principled_node: bpy.types.Node) -> None:
    colour = principled_node.inputs['Base Color'].default_value
    utils.set_principled_node(
        principled_node=principled_node,
        base_color=colour,
        metallic=0.0,
        specular=0.3,
        roughness=0.9,
    )


def set_principled_node_as_plane(principled_node: bpy.types.Node) -> None:

    utils.set_principled_node(
        principled_node=principled_node,
        base_color=(0.03, 0.03, 0.03, 0.03),
        metallic=0.0,
        specular=0.1,
        roughness=0.9,
    )


def set_principled_node_as_specular(principled_node: bpy.types.Node) -> None:
    utils.set_principled_node(
        principled_node=principled_node,
        base_color=(random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), random.uniform(0.0, 1.0),
                    random.uniform(0.0, 1.0)),
        metallic=0.9,
        specular=0.5,
        roughness=0.2,
        anisotropic=0.8
    )


def set_scene_objects() -> bpy.types.Object:
    objects = utils.generate_items()
    principles = []
    i = 0
    for item in objects:
        item.cycles_visibility.shadow = False
        mat = utils.add_material("Material_" + str(i), use_nodes=True, make_node_tree_empty=True)
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
        links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
        set_principled_node_as_specular(principled_node)
        principles.append(principled_node)
        i += 1
        item.data.materials.append(mat)

    for item in objects:
        scene = bpy.context.scene
        scene.collection.objects.link(item)
    current_object = utils.create_plane(size=50.0, name="Floor")
    mat = utils.add_material("Material_Plane", use_nodes=True, make_node_tree_empty=True)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    set_principled_node_as_plane(principled_node)
    links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
    current_object.data.materials.append(mat)

    bpy.ops.object.empty_add(location=(0.0, -0.75, 1.0))

    focus_target_1 = bpy.context.object

    bpy.ops.object.empty_add(location=(0.3, -0.75, 1.0))

    focus_target_2 = bpy.context.object
    return principles, objects, focus_target_1, focus_target_2


def set_diffuse(principles):

    for node in principles:
        set_principled_node_as_diffuse(node)


def get_calibration_matrix_K(cam_data):
    f_in_mm = cam_data.lens

    scene = bpy.context.scene

    resolution_x_in_px = scene.render.resolution_x

    resolution_y_in_px = scene.render.resolution_y

    scale = scene.render.resolution_percentage / 100

    sensor_width_in_mm = cam_data.sensor_width

    sensor_height_in_mm = cam_data.sensor_height

    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    if cam_data.sensor_fit == 'VERTICAL':
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else:
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0

    camera_matrix = Matrix(
        ((alpha_u, skew, u_0),
         (0, alpha_v, v_0),
         (0, 0, 1)))
    return np.asarray(camera_matrix)


def get_RT_matrix(cam):

    blender_to_cv = Matrix(
        ((1, 0, 0),
         (0, -1, 0),
         (0, 0, -1)))

    location, rotation = cam.matrix_world.decompose()[0:2]

    rotation_blender = rotation.to_matrix().transposed()

    location_blender = -1 * rotation_blender @ location

    rotation_cv = blender_to_cv @ rotation_blender
    location_cv = blender_to_cv @ location_blender

    # put into 3x4 matrix
    rt_matrix = Matrix((
        rotation_cv[0][:] + (location_cv[0],),
        rotation_cv[1][:] + (location_cv[1],),
        rotation_cv[2][:] + (location_cv[2],)
    ))
    return np.asarray(rt_matrix)


def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x


def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit


def generate_light(energy, name, light_type):

    light_data = bpy.data.lights.new(name=f'{name}_data', type=light_type)
    light_data.energy = energy

    # create new object with the light data block
    light_object = bpy.data.objects.new(name=f'{name}_object', object_data=light_data)

    # link light object
    bpy.context.collection.objects.link(light_object)

    # make it active
    bpy.context.view_layer.objects.active = light_object

    # change location
    light_object.location = (random.randint(-4, 4), random.randint(-4, 4), random.randint(9, 11))

    # update scene
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()

    return light_object


def write_matrices(camera_object, file_name, view):

    camera_matrix = get_calibration_matrix_K(camera_object.data)
    rt_matrix = get_RT_matrix(camera_object)

    if not os.path.exists(MATRIX_DIR):
        os.mkdir(MATRIX_DIR)

    k_file = open(f'{MATRIX_DIR}K_matrix_{file_name}{view}', 'wb')
    pkl.dump(camera_matrix, k_file)
    k_file.close()

    rt_file = open(f'{MATRIX_DIR}RT_matrix_{file_name}{view}', 'wb')
    pkl.dump(rt_matrix, rt_file)
    rt_file.close()


def get_item_locations(camera_object, objects_in_scene):
    rt_matrix = get_RT_matrix(camera_object)

    full_matrix = np.zeros((4, 4))
    full_matrix[:-1, :] = rt_matrix
    full_matrix[3][3] = 1

    item_locations = []
    for item in objects_in_scene:
        location, rotation = item.matrix_world.decompose()[0:2]

        loc = np.zeros(4)
        loc[:-1] = location
        loc[3] = 1

        item_location = full_matrix @ loc

        item_locations.append([item_location[1], item_location[0], item_location[2]])

    item_locations = np.asarray(item_locations)
    return np.asarray(item_locations)


def generate(resolution, sample_count, data_count):
    enable_gpus("CUDA")

    write_matrix = True

    if not os.path.exists(RENDER_DIR):
        os.mkdir(RENDER_DIR)

    if not os.path.exists(LOCATIONS_DIR):
        os.mkdir(LOCATIONS_DIR)

    for render_num in range(0, data_count):

        # Scene Building
        scene = bpy.data.scenes["Scene"]
        scene.render.resolution_x = 1024
        scene.render.resolution_y = 1024

        # Reset scene
        utils.clean_objects()

        # Set scene
        principles, objects_in_scene, focus_target_1, focus_target_2 = set_scene_objects()

        # Set lights
        light_object_1 = generate_light(energy=10000, name='light_1', light_type='POINT')

        light_object_2 = generate_light(energy=10000, name='light_2', light_type='POINT')

        # Set cameras
        bpy.ops.object.camera_add(location=(-1.5, -16.0, 40.0))
        camera_object_1 = bpy.context.object

        bpy.ops.object.camera_add(location=(-1.2, -16.0, 40.0))
        camera_object_2 = bpy.context.object

        utils.add_track_to_constraint(camera_object_1, focus_target_1)
        utils.set_camera_params(camera_object_1.data, focus_target_1, lens=85, fstop=0.5)

        utils.add_track_to_constraint(camera_object_2, focus_target_2)
        utils.set_camera_params(camera_object_2.data, focus_target_2, lens=85, fstop=0.5)

        file_name = str(render_num)

        utils.set_output_properties(scene, resolution, (SPECULAR_DIR + file_name + "_1"))
        utils.set_cycles_renderer(scene, camera_object_1, sample_count)

        # Render first specular view
        bpy.ops.render.render(write_still=True)

        utils.set_output_properties(scene, resolution, (SPECULAR_DIR + file_name + "_2"))
        utils.set_cycles_renderer(scene, camera_object_2, sample_count)

        # Render second specular view
        bpy.ops.render.render(write_still=True)

        set_diffuse(principles)

        light_object_1.hide_render = True
        light_object_2.hide_render = True

        # Generate light for diffuse
        light_object_3 = generate_light(energy=100, name='light_3', light_type='AREA')
        # light_data_3.use_shadow = False

        light_object_3.location = (0, 0, 10)
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()

        utils.set_output_properties(scene, resolution, (DIFFUSE_DIR + file_name + "_1"))
        utils.set_cycles_renderer(scene, camera_object_1, sample_count)

        # Render first diffuse view
        bpy.ops.render.render(write_still=True)

        if write_matrix:
            write_matrices(camera_object_1, file_name, '_1')

        utils.set_output_properties(scene, resolution, (DIFFUSE_DIR + file_name + "_2"))
        utils.set_cycles_renderer(scene, camera_object_2, sample_count)

        # Render second diffuse view
        bpy.ops.render.render(write_still=True)

        if write_matrix:
            write_matrices(camera_object_2, file_name, '_2')
            write_matrix = False

        item_locations = get_item_locations(camera_object_1, objects_in_scene)

        loc_file = open(f'{LOCATIONS_DIR}location_{file_name}', 'wb')
        pkl.dump(item_locations, loc_file)
        loc_file.close()


# Args
resolution_percentage = int(sys.argv[sys.argv.index('--') + 2])
num_samples = int(sys.argv[sys.argv.index('--') + 3])
num_data = int(sys.argv[sys.argv.index('--') + 4])
generate(resolution_percentage, num_samples, num_data)
