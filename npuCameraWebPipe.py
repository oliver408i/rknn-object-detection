# NPU Camera Streamer
from rknnlite.api import RKNNLite # type: ignore
import os, time, logging, argparse, sys, signal
import numpy as np
import cv2
import ppcb
from multiprocessing import Process, Queue, Manager

parser = argparse.ArgumentParser(description="RKNN Npu Object detection pipeline supporting YOLOv5 and YOLOv8.")
parser.add_argument('--model', type=str, required=True, help="Path to the RKNN model file (e.g., model.rknn)")
parser.add_argument('--type', type=str, choices=['yolov5', 'yolov8'], required=True, help="Model type (yolov5 or yolov8)")
parser.add_argument('--camera', type=int, default=0, help="Camera index")

args = parser.parse_args()

mfs = args.model
model_type = args.type

if (model_type == 'yolov5'):
    fd = ppcb.Yolov5.filter_detections
elif (model_type == 'yolov8'):
    fd = ppcb.Yolov8.filter_detections
else:
    raise ValueError("Model type must be either 'yolov5' or 'yolov8'")

sigTerm = False

def signal_handler(sig, frame):
    global sigTerm
    if sigTerm:
        return
    sigTerm = True
    logger = setupLogger("signal_handler")
    logger.warning("Closing all procs")
    os.system(f"killall -9 {sys.executable}")
    sys.exit(0)

class nparrayContainer:
    def __init__(self, data, shape, dtype):
        self.data = data
        self.shape = shape
        self.dtype = dtype
    
    def getNp(self):
        return np.frombuffer(self.data, dtype=self.dtype).reshape(self.shape)

class Detection:
    def __init__(self, image: nparrayContainer, frame_id: int, timeStamp: float):
        self.image = image
        self.frame_id = frame_id  # Add a unique ID for each frame
        self.detections = None
        self.timeStamp = timeStamp

def crop640(image):
    # Get the original image size
    # Get the dimensions of the image
    h, w = image.shape[:2]
    
    # Determine the center cropping size (square based on smallest dimension)
    crop_size = min(h, w)
    
    # Calculate the coordinates for the center cropping
    x_center = w // 2
    y_center = h // 2
    
    # Calculate the top-left corner of the cropping box
    x1 = x_center - crop_size // 2
    y1 = y_center - crop_size // 2
    
    # Crop the image (centered)
    cropped_image = image[y1:y1+crop_size, x1:x1+crop_size]
    
    # Resize the cropped image to 640x640
    final_image = cv2.resize(cropped_image, (640, 640))
    
    return final_image

def overwriteLast(lines=1):
    print("\033[A                             \033[A"*lines)

npuResolverCount = 4*3
postProcessorCount = 3

class ColoredFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[93;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    purple = "\x1b[35;20m"
    format = "%(asctime)s - %(name)s - "

    FORMATS = {
        logging.DEBUG: format + purple + "%(levelname)s - %(message)s" + reset,
        logging.INFO: format + grey + "%(levelname)s - %(message)s" + reset,
        logging.WARNING: format + yellow + "%(levelname)s - %(message)s" + reset,
        logging.ERROR: format + red + "%(levelname)s - %(message)s" + reset,
        logging.CRITICAL: format + bold_red + "%(levelname)s - %(message)s" + reset
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setupLogger(name):
    # Get a logger instance
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if (logger.hasHandlers()):
        logger.handlers.clear()
    
    console_handler = logging.StreamHandler()

    # Apply the custom formatter
    formatter = ColoredFormatter()
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    return logger

import time

def displayFrames(output_queue):
    last_displayed_id = -1  # Keep track of the last displayed frame ID
    logger = setupLogger("displayFrames")
    fps = 0
    frame_count = 0
    start_time = time.time()

    while True:
        if not output_queue:
            continue

        # Safely iterate over a copy of the output queue
        available_frame_ids = [task for task in list(output_queue) if task is not None]

        if not available_frame_ids:
            continue

        # Check if the next frame in sequence is available
        next_expected_id = last_displayed_id + 1

        for i in available_frame_ids:
            if i.frame_id >= next_expected_id:
                img = i.image.getNp()
                drawn_frame = ppcb.CommonOps.draw_boxes(img, i.detections)

                if drawn_frame is not None:
                    # Calculate FPS
                    frame_count += 1
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    if elapsed_time > 1:
                        fps = frame_count / elapsed_time
                        frame_count = 0
                        start_time = end_time

                    # Calculate latency in milliseconds
                    latency_ms = round((time.time() - i.timeStamp) * 1000)

                    # Display FPS and latency on the frame
                    cv2.putText(drawn_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(drawn_frame, f"Latency: {latency_ms} ms", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Show the frame
                    cv2.imshow("out", drawn_frame)
                    cv2.waitKey(1)
                    last_displayed_id = i.frame_id
                    output_queue[:] = []
                break


def npuResolver(input_queue, worker_id, post_processing_queue):
    logger = setupLogger("npuResolver " + str(worker_id))
    resolver = RKNNLite()
    #overwriteLast(1)
    ret = resolver.load_rknn(mfs)
    if ret != 0:
        logger.critical("Load failed")
        exit(ret)
    
    resolver.init_runtime(core_mask=[RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2][worker_id%3])
    #overwriteLast(5)
    #resolver.init_runtime(core_mask=RKNNLite.NPU_CORE_ALL)

    logger.debug("Process " + str(worker_id) + " loaded using NPU core #"+str(worker_id%3))

    while True:
        task = input_queue.get()
        if task is None or task.image is None:  # Check for the sentinel value
            break

        img = task.image.getNp()
        img = np.expand_dims(img, 0)
        outputs = resolver.inference(inputs=[img])
        task.detections = outputs
        post_processing_queue.put(task)  # Send raw output for post-processing

    resolver.release()
    logger.info("NPU resolver " + str(worker_id) + " closed")

def postProcessor(post_processing_queue, outputs_queue, wid):
    logger = setupLogger("postProcessor " + str(wid))
    logger.debug("Post-processing loaded on worker #" + str(wid))
    while True:
        output = post_processing_queue.get()
        if output is None:  # Check for the sentinel value
            break

        processed_output = fd(output.detections, threshold=0.3)
        output.detections = processed_output

        if len(outputs_queue) > 10:
            dropped_frame = outputs_queue.pop(0)  # Drop the oldest frame
            logger.warning(f"Dropped the oldest output with frame ID: {dropped_frame.frame_id}")

        outputs_queue.append(output)

    logger.info("Post-processing process #" + str(wid) + " closed")


def cameraStreamer(input_queue):
    logger = setupLogger("cameraStreamer")
    camera = cv2.VideoCapture(args.camera)
    frame_id = 0  # Initialize the frame ID counter
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    while(True):
        ret, frame = camera.read()
        if ret:
            frame = crop640(frame)
            input_queue.put(Detection(nparrayContainer(frame.tobytes(), frame.shape, frame.dtype), frame_id, time.time()))
            frame_id += 1  # Increment the frame ID for each new frame
        else:
            logger.critical("Camera streamer failed to read frame")

    


logger = setupLogger("main")



logger.info("RKNN NPU Pipeline running on " + str(npuResolverCount) + " NPU workers and " + str(postProcessorCount) + " post-processing workers")
logger.info("Model: " + mfs)
logger.info("Model type: " + model_type)

if cv2.ocl.haveOpenCL():
    cv2.ocl.setUseOpenCL(True)
    logger.debug("OpenCL enabled for GPU image processing")
else:
    logger.warn("OpenCL not supported. Fallback to CPU image processing")

logger.info("Env resolved")


folderOfImages = "images"

logger.info("-----------------------")


with Manager() as manager:
    post_processing_queue = Queue()
    outputs_queue = manager.list()
    input_queue = Queue()

    # Register the signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    processes = []

    c = Process(target=cameraStreamer, args=(input_queue, ))
    c.start()

    # Start NPU resolver processes
    for i in range(npuResolverCount):
        p = Process(target=npuResolver, args=(input_queue, i, post_processing_queue))
        p.start()
        processes.append(p)

    # Start the post-processing processes
    for i in range(postProcessorCount):
        p = Process(target=postProcessor, args=(post_processing_queue, outputs_queue, i))
        p.start()
        processes.append(p)

    # Start image display
    w = Process(target=displayFrames, args=(outputs_queue,))
    w.start()
    processes.append(w)

    

    # Wait for all NPU resolver processes to finish
    for p in processes[:npuResolverCount]:
        p.join()

    # Wait for the post-processing and printer processes to finish
    for p in processes[npuResolverCount:]:
        p.join()



    

