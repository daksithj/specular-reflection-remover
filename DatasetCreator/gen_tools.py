import os
import subprocess
import shutil


def start_generating(dataset_name, model_list, data_num, background=None, objects=None, external=True):

    if external:
        script = "DatasetCreator/synthetic_gen.py"
        assets_base = f"DatasetCreator/assets"
        asset_dir = f"{assets_base}/{dataset_name}"
    else:
        script = "synthetic_gen.py"
        assets_base = f'assets'
        asset_dir = f"{assets_base}/{dataset_name}"

    if not os.path.exists(assets_base):
        os.mkdir(assets_base)

    if not os.path.exists(asset_dir):
        os.mkdir(asset_dir)

    for file in model_list:
        shutil.copy(file, asset_dir)

    gen_cmd = f'blender --background --python {script} -- {dataset_name} 50 120 {data_num}'

    if background is not None:
        gen_cmd += f' --background {background}'

    if objects is not None:
        gen_cmd += f' --object {objects}'

    process = subprocess.Popen(gen_cmd, stdout=subprocess.PIPE)

    return process
    # while True:
    #     line = process.stdout.readline()
    #     if not line:
    #         break
    #     if "Generated output" in str(line):
    #         number = str(line).split('_')[1]
    #         number = int(number)
    #         print(number)
    #
    # shutil.rmtree(asset_dir)