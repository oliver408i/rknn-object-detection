# rknn-object-detection
python scripts for rknn object detection for Orange pi 5. Based on yolov5 object detection. Runs on all 3 NPU cores, averages ~14 FPS, ~300 ms latency streaming camera. ~30 FPS without streaming camera.

# How to make a model
### Setup yolov5 toolkit
Python3.12 required
1. Clone the [yolov5 github repo](https://github.com/ultralytics/yolov5)
2. Setup a venv to use with the toolkit
3. Install all the requirements `pip install -r requirements.txt`
4. Verify that it works (or just wait until you obtain the dataset to train)
### Obtain dataset
You can manually make a dataset, or use roboflow. Make sure the image dimensions match how you are going to train it
1. Get the correct dataset format downloaded and in the same folder as the yolov5 toolkit
2. Find the `.yaml` file for that dataset. You might want to rename it.
3. In that yaml file, update the paths for the `train` and `valid` folders so that they are correct
### Train 
You don't need a x86 linux machine for this setup. Use the most powerful machine you have for this. Recommended to use a good GPU such as the RTX seris or Apple M1,M2. Note that for CUDA, make sure you have the correct torch version install (you might need to manually update your CUDA version and/or reinstall pytorch with the current CUDA version support enabled).
1. Run the following command (make sure you are using your inputs. For testing, do a 3 epoch training first) `python train.py --img 640 --epochs 30 --data data.yaml --weights yolov5.pt`
2. If you are on a silicon mac using m1 or m2 (m3 not supported yet), add `--device mps`. If you are on a cuda device (Nvidia GPU) add `--device 0`. If you encounter issues, remove the flag to train on cpu
3. Once the training is finished, verify that you have a `runs/train/exp/weights/best.pt` file. Note that the `exp` folder may become `exp2` or `exp3` if you ran multiple trainings. The highest number is the latest training
4. You can upload the model to roboflow using `upload.py`. Make sure to change the version and expNum variables. After uploading, use the visualize tool on roboflow to test your model
### Convert to ONNX
1. Once training is finished, run `python export.py --weights /runs/train/exp/weights/best.pt --include onnx`
2. Find the output onnx file in the same directory was your `best.pt` file
### Converting to RKNN
1. Setup a linux x86 device with python3.10. Install the RKNN toolkit python package from [here](https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit2/packages)
2. Upload your `best.onnx` file to that device
3. Run `convert.py` to get the output `model.rknn` file
### Orange pi
Written for Orange Pi 5 Pro (RK3588 CPU & NPU). **Must use a powerful heat sink or small fan!**
1. Upload your `model.rknn` file, as well as `npuCameraPipe.py` and `ppcb.py`.
2. Run with `python3 npuCameraPipe.py --model model.rknn --type yolov5`
3. ^C to exit
### Yolov8
This project now supports yolov8 detection models. Simply convert them using the same steps and upload them. When running, use `--type yolov8` to switch to the yolov8 post processing pipeline. Yolov8 segmentation support is planned for the future (yolov5 segmentation is not considered at this moment).
## Notes on the code
The only code here that is completely new is the `pp.py` and two pipeline programs. All the converting code is based off existing RKNN api examples and resources. The `upload.py` is just a short script to upload the finished model and I don't consider it a full program.  
The whole RKNN toolkit is not well documented (and also not well written) which made this whole thing a complete mess to make. It did work out in the end, which is why this is here now, but it took quite a lot of work.

## Code Structure
Final camera stream system. Each box is a process. WIP explanation
![NPU pipeline(1)](https://github.com/user-attachments/assets/da31de91-0ed7-4f7e-b67f-7f0b79e849fb)


test by rian :D
