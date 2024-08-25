from roboflow import Roboflow


import shutil
import os

expNum = 2
v = 5

source_dir = f"runs/train/exp{expNum}/weights"
destination_dir = f"runs/train/exp{expNum}"

for filename in os.listdir(source_dir):
    if filename.endswith(".pt"):
        shutil.copy(os.path.join(source_dir, filename), destination_dir)

rf = Roboflow(api_key="your key") # To see your key, export any dataset and click on the python tab
project = rf.workspace("pixel-data").project("pixel-detector")
version = project.version(v)

version.deploy("yolov5",f"runs/train/exp{expNum}","best.pt")
