
from rknnlite.api import RKNNLite # type: ignore
import os, time, logging, sys, signal, re, json
import numpy as np
import cv2
import ppcb
from multiprocessing import Process, Queue, Manager
from threading import Thread
from bottle import Bottle, request, run # type: ignore
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler

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
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if (logger.hasHandlers()):
        logger.handlers.clear()
    console_handler = logging.StreamHandler()
    formatter = ColoredFormatter()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

logger = setupLogger("argParser")

# Initialize default values
use_webserver = False
model_type = None
model_path = None
camera_index = 0
webserver_port = 8001

# Define a regex pattern to detect combined flags (e.g., -wtv5m)
combined_pattern = re.compile(r'^-w?(t(v5|v8))?(m.+)?$')

# Loop through the command-line arguments
i = 1  # Start after the script name (sys.argv[0] is the script name)
while i < len(sys.argv):
    arg = sys.argv[i]
    match = combined_pattern.match(arg)
    if match:
        # Handle the combined flag
        if 'w' in arg:
            use_webserver = True
        if match.group(1):  # Extract model type (e.g., tv5 or tv8)
            model_type = 'yolov5' if match.group(2) == 'v5' else 'yolov8'
        if match.group(3):  # Extract model path if it's combined (e.g., -mmodel.rknn)
            model_path = match.group(3)[1:]  # Remove the 'm' prefix
    else:
        # Handle standalone arguments like -m, -t, -c, -wp
        if arg == '-m':
            i += 1
            if i < len(sys.argv):
                model_path = sys.argv[i]
            else:
                logger.critical("Error: Model path is required after -m.")
                sys.exit(1)
        elif arg.startswith('-m') and len(arg) > 2:
            model_path = arg[2:]
        elif arg.startswith('-t') and len(arg) > 2:
            if 'v5' in arg:
                model_type = 'yolov5'
            elif 'v8' in arg:
                model_type = 'yolov8'
        elif arg == '-t':
            i += 1
            if i < len(sys.argv):
                if sys.argv[i] == 'yolov5' or sys.argv[i] == 'yolov8':
                    model_type = sys.argv[i]
                else:
                    logger.critical("Error: Invalid model type. Use 'yolov5' or 'yolov8'.")
                    sys.exit(1)
            else:
                logger.critical("Error: Model type is required after -t.")
                sys.exit(1)
        elif arg.startswith('-c'):
            try:
                camera_index = int(arg[2:])
            except ValueError:
                logger.critical("Invalid camera index provided.")
                sys.exit(1)
        elif arg.startswith('-wp'):
            try:
                webserver_port = int(arg[3:])
            except ValueError:
                logger.critical("Invalid webserver port provided.")
                sys.exit(1)
    i += 1

if not model_type:
    model_type = "yolov8"

if not model_path:
    model_path = "modeltest.rknn"

logger.info("Model path: " + model_path)
mfs = model_path

if (model_type == 'yolov5'):
    fd = ppcb.Yolov5.filter_detections
elif (model_type == 'yolov8'):
    fd = ppcb.Yolov8.filter_detections
else:
    raise ValueError("Model type must be either 'yolov5' or 'yolov8'")

sigTerm = False

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy arrays to lists
        return super(NumpyEncoder, self).default(obj)

def webServer(outputs_queue, ldf):
    logger = setupLogger("webServer")
    app = Bottle()
    @app.route('/ws')
    def serve_ws():
        logger.debug("New connection")
        ws = request.environ.get('wsgi.websocket')
        lastldf = None
        if not ws:
            logger.warn("Invalid connection: Not ws")
            return 'Expected WebSocket request.'
        while True:
            if not 'det' in ldf or not ldf['det']:
                continue
            if lastldf == ldf['det']:
                continue
            ws.send(json.dumps(ldf['det'], cls=NumpyEncoder))
            lastldf = ldf['det']
    server = pywsgi.WSGIServer(('100.64.0.7', webserver_port), app, handler_class=WebSocketHandler)
    server.serve_forever()

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
    return cv2.resize(image, (640, 640))

def overwriteLast(lines=1):
    print("\033[A                             \033[A"*lines)

npuResolverCount = 4*3  # 12 NPU workers
postProcessorCount = 3  # 3 post-processing workers
resizeWorkerCount = 4  # 4 resizing workers

def displayFrames(output_queue, ldf):
    last_displayed_id = -1  # Keep track of the last displayed frame ID
    logger = setupLogger("displayFrames")
    fps = 0
    frame_count = 0
    start_time = time.time()

    def resize_and_display_frame(drawn_frame):
        resized_frame = cv2.resize(drawn_frame, (1280, 720))
        cv2.imshow("out", resized_frame)
        cv2.waitKey(1)

    while True:
        if not output_queue:
            continue

        available_frame_ids = [task for task in list(output_queue) if task is not None]

        if not available_frame_ids:
            continue

        next_expected_id = last_displayed_id + 1

        for i in available_frame_ids:
            if i.frame_id >= next_expected_id:
                img = i.image.getNp()
                drawn_frame = ppcb.CommonOps.draw_boxes(img, i.detections)

                if drawn_frame is not None:
                    frame_count += 1
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    if elapsed_time > 1:
                        fps = frame_count / elapsed_time
                        frame_count = 0
                        start_time = end_time

                    if i.detections:
                        ldf["det"] = {
                            "frame_id": i.frame_id,
                            "timeStamp": i.timeStamp,
                            "detections": i.detections
                        }

                    latency_ms = round((time.time() - i.timeStamp) * 1000)

                    cv2.putText(drawn_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(drawn_frame, f"Latency: {latency_ms} ms", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Use main thread to display the frame instead of a new thread
                    resize_and_display_frame(drawn_frame)

                    last_displayed_id = i.frame_id
                    output_queue[:] = []
                break

def npuResolver(input_queue, worker_id, post_processing_queue):
    logger = setupLogger("npuResolver " + str(worker_id))
    resolver = RKNNLite()
    ret = resolver.load_rknn(mfs)
    if ret != 0:
        logger.critical("Load failed")
        exit(ret)
    resolver.init_runtime(core_mask=[RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2][worker_id % 3])
    logger.debug("Process " + str(worker_id) + " loaded using NPU core #" + str(worker_id % 3))
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

def cameraStreamer(raw_frame_queue):
    logger = setupLogger("cameraStreamer")
    camera = cv2.VideoCapture(camera_index)
    frame_id = 0  # Initialize the frame ID counter
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    logger.debug("Starting camera stream")
    while(True):
        ret, frame = camera.read()
        if ret:
            #logger.debug(f"Read frame {frame_id}")
            raw_frame_queue.put((frame, frame_id, time.time()))
            frame_id += 1  # Increment the frame ID for each new frame
        else:
            logger.critical("Camera streamer failed to read frame")
            time.sleep(0.5)
            logger.critical("Unable to continue, causing stack crash...")
            signal_handler(None, None)

def resizeWorker(raw_frame_queue, resized_frame_queue, worker_id):
    logger = setupLogger("resizeWorker " + str(worker_id))
    logger.debug("Resizing worker started")
    while True:
        frame_data = raw_frame_queue.get()
        if frame_data is None:  # Check for the sentinel value
            break
        raw_frame, frame_id, timestamp = frame_data
        resized_frame = crop640(raw_frame)
        resized_frame_queue.put(Detection(nparrayContainer(resized_frame.tobytes(), resized_frame.shape, resized_frame.dtype), frame_id, timestamp))
        #logger.debug(f"Resized frame {frame_id} by worker {worker_id}")

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
    raw_frame_queue = Queue()  # New queue for raw frames
    ldf = manager.dict()

    # Register the signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    processes = []

    # Start the raw frame reader
    c = Process(target=cameraStreamer, args=(raw_frame_queue,))
    c.start()

    # Start the resize workers
    for i in range(resizeWorkerCount):
        r = Process(target=resizeWorker, args=(raw_frame_queue, input_queue, i))
        r.start()
        processes.append(r)

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
    w = Process(target=displayFrames, args=(outputs_queue, ldf))
    w.start()
    processes.append(w)

    if use_webserver:
        ws = Process(target=webServer, args=(outputs_queue, ldf))
        ws.start()
        processes.append(ws)

    c.join()
    for r in processes[:resizeWorkerCount]:
        r.join()

    # Wait for all NPU resolver processes to finish
    for p in processes[resizeWorkerCount:resizeWorkerCount + npuResolverCount]:
        p.join()

    # Wait for the post-processing and printer processes to finish
    for p in processes[resizeWorkerCount + npuResolverCount:]:
        p.join()