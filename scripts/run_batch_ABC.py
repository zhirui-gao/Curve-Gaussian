import os
import subprocess
directory = '/media/gzr/955be20b-af2b-4597-83f8-8585ff878672/ABC_dataset/ABC-NEF_Edge/data'
objects = sorted([f.name for f in os.scandir(directory) if f.is_dir()])
exp_name = 'output'
command_template = "python train.py -s {input_dir} -m {exp_name}/{object_name} --quiet -r 2 --eval" \
                   " --test_iterations 10000 --save_iterations 10000 --port 6609"


for object_name in objects:
    input_dir = os.path.join(directory, object_name)
    print(f"Running command: {object_name}")
    command = command_template.format(input_dir=input_dir, exp_name=exp_name,object_name=object_name)

    print(command)
    if os.path.exists(os.path.join(exp_name, object_name, 'parametric_edges.json')):
        continue

    subprocess.run(command, shell=True)
 

