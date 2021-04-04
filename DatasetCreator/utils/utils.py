import bpy
import math
from typing import Tuple
from utils.node import arrange_nodes

################################################################################
# Scene
################################################################################


def build_rgb_background(world: bpy.types.World,
                         rgb: Tuple[float, float, float, float] = (0.9, 0.9, 0.9, 1.0),
                         strength: float = 1.0) -> None:
    world.use_nodes = True
    node_tree = world.node_tree

    rgb_node = node_tree.nodes.new(type="ShaderNodeRGB")
    rgb_node.outputs["Color"].default_value = rgb

    node_tree.nodes["Background"].inputs["Strength"].default_value = strength

    node_tree.links.new(rgb_node.outputs["Color"], node_tree.nodes["Background"].inputs["Color"])

    arrange_nodes(node_tree)




def set_output_properties(scene: bpy.types.Scene, resolution_percentage: int = 100, output_file_path: str = "") -> None:
    scene.render.resolution_percentage = resolution_percentage

    if output_file_path:
        scene.render.filepath = output_file_path


def set_cycles_renderer(scene: bpy.types.Scene,
                        camera_object: bpy.types.Object,
                        num_samples: int,
                        engine: str = 'CYCLES',
                        use_denoising: bool = True,
                        use_motion_blur: bool = False,
                        use_transparent_bg: bool = False) -> None:
    scene.camera = camera_object

    scene.render.image_settings.file_format = 'PNG'
    scene.render.engine = engine
    scene.render.use_motion_blur = use_motion_blur

    scene.render.film_transparent = use_transparent_bg
    scene.view_layers[0].cycles.use_denoising = use_denoising

    scene.cycles.samples = num_samples


################################################################################
# Constraints
################################################################################


def add_track_to_constraint(camera_object: bpy.types.Object, track_to_target_object: bpy.types.Object) -> None:
    constraint = camera_object.constraints.new(type='LOCKED_TRACK')
    constraint.target = track_to_target_object
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.lock_axis = 'LOCK_X'


################################################################################
# Misc.
################################################################################


def clean_objects() -> None:
    for item in bpy.data.objects:
        bpy.data.objects.remove(item)
