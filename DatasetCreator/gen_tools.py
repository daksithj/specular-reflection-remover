import os
import subprocess
import shutil


def start_generating(dataset_name, model_list, data_num, external=True):

    if external:
        script = "DatasetCreator/synthetic_gen.py"
        asset_dir = f"DatasetCreator/assets/{dataset_name}"
    else:
        script = "synthetic_gen.py"
        asset_dir = f"assets/{dataset_name}"

    if not os.path.exists(asset_dir):
        os.mkdir(asset_dir)

    for file in model_list:
        shutil.copy(file, asset_dir)

    process = subprocess.Popen(f'blender --background --python {script} -- {dataset_name} 50 120 {data_num}',
                               stdout=subprocess.PIPE)

    while True:
        line = process.stdout.readline()
        if not line:
            break
        if "Generated output" in str(line):
            number = str(line).split('_')[1]
            number = int(number)
            print(number)

    shutil.rmtree(asset_dir)