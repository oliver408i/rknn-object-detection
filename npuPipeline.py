# Run on orange pi. sends out detections to console

from rknnlite.api import RKNNLite
import os, time, logging
import numpy as np
import cv2, pp
from multiprocessing import Process, Queue

logging.basicConfig(level=logging.DEBUG)

mfs = "model.rknn"
folderOfImages = "images"

npuResolverCount = 4*3
postProcessorCount = 5

def npuResolver(input_queue, worker_id, post_processing_queue):
    logger = logging.getLogger("npuResolver " + str(worker_id))
    resolver = RKNNLite()
    ret = resolver.load_rknn(mfs)
    if ret != 0:
        logger.critical("Load failed")
        exit(ret)
    
    resolver.init_runtime(core_mask=[RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2][worker_id%3])
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
    logger = logging.getLogger("postProcessor " + str(wid))
    logger.debug("Post-processing loaded on worker #" + str(wid))
    while True:
        output = post_processing_queue.get()
        if output is None:  # Check for the sentinel value
            break

        processed_output = pp.filter_detections(output, threshold=0.3)
        outputs_queue.put(processed_output)  # Send processed output to printer
    logger.info("Post-processing process #" + str(wid) + " closed")

def outputPrinter(outputs_queue):
    logger = logging.getLogger("outputPrinter")
    while True:
        task = outputs_queue.get()
        if task is None:  # Check for the sentinel value
            break
        logger.info(task)
    logger.info("Output printer process: closed")


logger = logging.getLogger("main")


logger.info("Env resolved")

logger.info("-----------------------")

startTime = time.time()
n = 0

input_queue = Queue()
post_processing_queue = Queue()
outputs_queue = Queue()

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

# Start the output printer process
o = Process(target=outputPrinter, args=(outputs_queue,))
o.start()

# Wait for all NPU resolver processes to finish
for p in processes[:npuResolverCount]:
    p.join()

# Send sentinel values to signal the end of post-processing and printing
for i in range(postProcessorCount):
    post_processing_queue.put(None)
outputs_queue.put(None)

waitTime = time.time()

# Wait for the post-processing and printer processes to finish
for p in processes[npuResolverCount:]:
    p.join()

o.join()

ppDelay = time.time() - waitTime
logger.debug("-----------------------")
logger.debug("Postprocessing delay total:"+str(ppDelay))
logger.debug("Postprocessing delay per image avg:"+str(ppDelay/n))
logger.debug("-----------------------")
o.join()


logger.info("Main process: done")

endTime = time.time()
totalTime = endTime - startTime
timePerImage = totalTime / n
fps = 1 / timePerImage
logger.info("-----------------------")
logger.info("Total time taken: "+str(totalTime))
logger.info("Time per image: "+str(timePerImage))
logger.info("FPS: "+str(fps))
