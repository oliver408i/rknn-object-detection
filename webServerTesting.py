from rknnlite.api import RKNNLite # type: ignore
import os, time, logging, sys
import numpy as np
import cv2, pp # type: ignore
from multiprocessing import Process, Queue, Manager


mfs = "model.rknn"

npuResolverCount = 4*3
postProcessorCount = 5

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

def webServer(output_queue):
    logger = setupLogger("webSocketServer")

    from bottle import Bottle, request, run # type: ignore
    
    app = Bottle()

    @app.route('/number')
    def serve_number():
        shared_list = app.config['shared_list']
        if shared_list:
            latest_number = shared_list[-1]  # Get the latest number
            return {'number': latest_number}
        else:
            return {'number': 'No number available'}, 204
    app.config['shared_list'] = output_queue
    logger.info("Starting server")
    
    while output_queue and output_queue[-1] is not None:
       run(app, host='0.0.0.0', port=8080,quiet=True)


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
        if task is None:  # Check for the sentinel value
            break
        img = cv2.imread(os.path.join(folderOfImages, task))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = np.expand_dims(img, 0)
        outputs = resolver.inference(inputs=[img])
        post_processing_queue.put(outputs)  # Send raw output for post-processing
        #time.sleep(0.00001)

    resolver.release()
    logger.info("NPU resolver " + str(worker_id) + " closed")

def postProcessor(post_processing_queue, outputs_queue, wid):
    logger = setupLogger("postProcessor " + str(wid))
    logger.debug("Post-processing loaded on worker #" + str(wid))
    while True:
        output = post_processing_queue.get()
        if output is None:  # Check for the sentinel value
            break

        processed_output = pp.filter_detections(output, threshold=0.3)
        if len(outputs_queue) > 10:
            outputs_queue.pop(0)
            logger.warning("Dropped an output")
        outputs_queue.append(processed_output)

    logger.info("Post-processing process #" + str(wid) + " closed")


logger = setupLogger("main")

if cv2.ocl.haveOpenCL():
    cv2.ocl.setUseOpenCL(True)
    logger.info("OpenCL enabled")


logger.info("Env resolved")

folderOfImages = "images"

logger.info("-----------------------")


n = 0

input_queue = Queue()
post_processing_queue = Queue()
with Manager() as manager:
    outputs_queue = manager.list()

    processes = []

    # Optimized Enqueue of tasks
    for filename in os.listdir(folderOfImages):
        if filename.endswith(".jpg"):

            input_queue.put(filename)  # Directly enqueue the image
            n += 1

    # Add sentinel values to signal the end of tasks
    for i in range(npuResolverCount):
        input_queue.put(None)

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

    # Start the web server
    w = Process(target=webServer, args=(outputs_queue,))
    w.start()

    startTime = time.time()

    # Wait for all NPU resolver processes to finish
    for p in processes[:npuResolverCount]:
        p.join()

    # Send sentinel values to signal the end of post-processing and printing
    for i in range(postProcessorCount):
        post_processing_queue.put(None)
    outputs_queue.append(None)
    w.join()

    waitTime = time.time()

    # Wait for the post-processing and printer processes to finish
    for p in processes[npuResolverCount:]:
        p.join()

    ppDelay = time.time() - waitTime
    logger.debug("-----------------------")
    logger.debug("Postprocessing delay total:"+str(ppDelay))
    logger.debug("Postprocessing delay per image avg:"+str(ppDelay/n))
    logger.debug("-----------------------")


    logger.info("Main process: done")

    endTime = time.time()
    totalTime = endTime - startTime
    timePerImage = totalTime / n
    fps = 1 / timePerImage
    logger.info("-----------------------")
    logger.info("Total time taken: "+str(totalTime))
    logger.info("Time per image: "+str(timePerImage))
    logger.info("FPS: "+str(fps))
