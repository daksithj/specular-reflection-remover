import bpy
import os
import random
from typing import Tuple, Optional



def create_plane(location: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 size: float = 2.0,
                 name: Optional[str] = None) -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=size, location=location, rotation=rotation)

    current_object = bpy.context.object

    if name is not None:
        current_object.name = name

    return current_object



def create_item(objects, location, num, subdivision_level: int = 2):
    directory = "C:/Users/Daksith/Documents/Uni/Final year project/blender-cli-rendering-master/assets/models"
    file_name = random.choice(os.listdir(directory))
    file = directory + '/' + file_name
    bpy.ops.import_mesh.ply(filepath=file)

    current_object = bpy.context.object
    current_object.name = "Model_obj_" + str(num)
    loc = location.pop()
    current_object.location = loc
    #current_object.dimensions = (random.randint(2, 3), random.randint(2, 3), random.randint(2, 3))
    # Spoon
    # current_object.dimensions = (0.75, 0.5, 5)
    #current_object.dimensions = (3, 3, 3)
    dim = current_object.dimensions
    fac = 4 / max(dim)
    current_object.scale = (fac, fac, fac)
    current_object.rotation_euler = (random.uniform(-3.14159, 3.14159), random.uniform(-3.14159, 3.14159),
                                     random.uniform(-3.14159, 3.14159))
    # add_subdivision_surface_modifier(current_object, subdivision_level)
    objects.append(current_object)

    return objects


def generate_items():
    og_loc_x = -5.0
    og_loc_y = -2.5
    og_loc_z = 1.5

    location = []
    loc_x = og_loc_x
    loc_y = og_loc_y
    for j in range(2):
        for k in range(3):
            location.append((loc_x, loc_y, og_loc_z))
            loc_x = loc_x + 13 / 3
        loc_x = og_loc_x
        loc_y = loc_y + 8 / 2
    # location.pop(0)
    random.shuffle(location)

    objects = []

    num_ob = 6
    for i in range(num_ob):
        objects = create_item(objects, location, i)

    return objects