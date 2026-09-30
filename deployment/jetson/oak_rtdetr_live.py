import time
import cv2
import depthai as dai

from rdd_runtime import detections, annotate
from rdd_raw_runtime import RawTensorRTEngine

ENGINE = "models/nano_raw_fp16.engine"
CONF = 0.60

pipeline = dai.Pipeline()
cam = pipeline.createColorCamera()
cam.setPreviewSize(640, 480)
cam.setFps(5)

xout = pipeline.createXLinkOut()
xout.setStreamName("rgb")
cam.preview.link(xout.input)

runner = RawTensorRTEngine(ENGINE)

try:
    with dai.Device(pipeline, usb2Mode=True) as device:
        q = device.getOutputQueue("rgb", maxSize=1, blocking=False)

        print("OAK-D + RT-DETR READY")

        while True:
            packet = q.get()
            frame = packet.getFrame()

            start = time.perf_counter()
            items = detections(runner.predict(frame), CONF)
            fps = 1.0 / max(time.perf_counter() - start, 1e-9)

            rendered = annotate(frame, items, fps)

            cv2.imshow("OAK-D RT-DETR - q to quit", rendered)

            if cv2.waitKey(1) & 0xff == ord("q"):
                break
finally:
    runner.close()
    cv2.destroyAllWindows()
