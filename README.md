# rknn-object-detection
python scripts for rknn object detection for Orange pi 5. Based on yolov5 object detection. Runs on all 3 NPU cores

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
### Run on Orange Pi
1. Upload `pp.py` and `npuPipeline.py` as well as `model.rknn` to the orange pi.
2. You can use `npuPipeline.py`, which will get all detections of every `.jpeg` image in the `images` folder. You can change this in the code. You can also change the output to go wherever you need it to go
Camera streaming is still WIP and to be tested in Tuesday
### Running yolov8
Yolov8 detection models are only supported on the multi model NPU pipeline. To use that, do `python npuPipelineWebMultiModel.py --model model.rknn --type yolov8`. Type can be yolov5 or yolov8. Make sure to copy `ppv5.py` and `ppv8.py` to use this pipeline. You don't need to copy `pp.py` as that is the legacy post processor functions for the old `npuPipeline.py`. Yolov8 models should be converted to RKNN in the same way. No support for Yolov8 segmentation models yet as I try to figure out how the output is formatted (it is quite complex).
### Private usecase
- `webServerTesting.py` opens a Bottle webserver on 8080 that serves the latest detection result directly from the npu pipeline. Only the latest result is served and the oldest results are deleted to avoid a memory leak in the pipeline.
- The web server is meant for other devices to easily access detection results
- It is current just a POC and closes itself after all images are processed, but once camera streaming is implemented, it will facilitate a (hopefully) reliable and simple way to send results elsewhere
## Notes on the code
The only code here that is completely new is the `pp.py` and two pipeline programs. All the converting code is based off existing RKNN api examples and resources. The `upload.py` is just a short script to upload the finished model and I don't consider it a full program.  
The whole RKNN toolkit is not well documented (and also not well written) which made this whole thing a complete mess to make. It did work out in the end, which is why this is here now, but it took quite a lot of work.
