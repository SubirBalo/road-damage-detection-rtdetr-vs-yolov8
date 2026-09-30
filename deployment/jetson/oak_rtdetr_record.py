import time
import cv2
import depthai as dai

from rdd_runtime import detections, annotate
from rdd_raw_runtime import RawTensorRTEngine

ENGINE = "models/nano_raw_fp16.engine"
CONF = 0.60
OUTPUT = "results/oak_live/oak_rtdetr_demo.mp4"

pipeline = dai.Pipeline()

cam = pipeline.createColorCamera()
cam.setPreviewSize(640, 480)
cam.setFps(5)

xout = pipeline.createXLinkOut()
xout.setStreamName("rgb")
cam.preview.link(xout.input)

runner = RawTensorRTEngine(ENGINE)

writer = cv2.VideoWriter(
    OUTPUT,
    cv2.VideoWriter_fourcc(*"mp4v"),
    5.0,
    (640, 480)
)

if not writer.isOpened():
    raise RuntimeError("Video writer failed")

frames = 0
started = time.perf_counter()

try:
    with dai.Device(pipeline, usb2Mode=True) as device:

        q = device.getOutputQueue(
            "rgb",
            maxSize=1,
            blocking=False
        )

        print("OAK-D + RT-DETR RECORDING")
        print("Saving:", OUTPUT)
        print("Press q to stop")

        while True:
            packet = q.get()
            frame = packet.getFrame()

            tick = time.perf_counter()

            items = detections(
                runner.predict(frame),
                CONF
            )

            fps = 1.0 / max(
                time.perf_counter() - tick,
                1e-9
            )

            rendered = annotate(frame, items, fps)

            writer.write(rendered)
            frames += 1

            cv2.imshow(
                "OAK-D RT-DETR RECORDING - q to quit",
                rendered
            )

            if cv2.waitKey(1) & 0xff == ord("q"):
                break

finally:
    elapsed = time.perf_counter() - started
    writer.release()
    runner.close()
    cv2.destroyAllWindows()

    print("Frames:", frames)
    print("Seconds:", elapsed)

    if elapsed > 0:
        print("Average FPS:", frames / elapsed)

    print("Saved:", OUTPUT)
