import numpy as np  # numpy - manipulate the packet data returned by depthai
import cv2  # opencv - display the video stream
import depthai  # depthai - access the camera and its data packets
import blobconverter  # blobconverter - compile and download MyriadX neural network blobs


# Pipeline tells DepthAI what operations to perform when running - you define all of the resources used and flows here
pipeline = depthai.Pipeline()

# First, we want the Color camera as the output
cam_rgb = pipeline.create(depthai.node.ColorCamera)
cam_rgb.setPreviewSize(300, 300) # 300x300 will be the preview frame size, available as 'preview' output of the node
cam_rgb.setInterleaved(False)

# Next, we want a neural network that will produce the detections
detection_nn = pipeline.create(depthai.node.MobileNetDetectionNetwork)

# Blob is the Neural Network file, compiled for MyriadX. It contains both the definition and weights of the model
# We're using a blobconverter tool to retreive the MobileNetSSD blob automatically from OpenVINO Model Zoo
detection_nn.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))

# Next, we filter out the detections that are below a confidence threshold. Confidence can be anywhere between <0..1>
detection_nn.setConfidenceThreshold(0.5)

# Next, we link the camera 'preview' output to the neural network detection input, so that it can produce detections
cam_rgb.preview.link(detection_nn.input)

# XLinkOut is a "way out" from the device. Any data you want to transfer to host need to be send via XLink
xout_rgb = pipeline.create(depthai.node.XLinkOut)
# For the rgb camera output, we want the XLink stream to be named "rgb"
xout_rgb.setStreamName("rgb")
# Linking camera preview to XLink input, so that the frames will be sent to host
cam_rgb.preview.link(xout_rgb.input)

# The same XLinkOut mechanism will be used to receive nn results
xout_nn = pipeline.create(depthai.node.XLinkOut)
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# Pipeline is now finished, and we need to find an available device to run our pipeline
# we are using context manager here that will dispose the device after we stop using it
with depthai.Device(pipeline) as device:
    # From this point, the Device will be in "running" mode and will start sending data via XLink

    # To consume the device results, we get two output queues from the device, with stream names we assigned earlier
    # Qurilma natijalarini iste'mol qilish uchun biz qurilmadan oldin tayinlangan oqim nomlari bilan ikkita chiqish navbatini olamiz
    q_rgb = device.getOutputQueue("rgb")
    q_nn = device.getOutputQueue("nn")

    # Here, some of the default values are defined. Frame will be an image from "rgb" stream, detections will contain nn results
    # Bu yerda bir necha standart qiymatlar belgilangan. Kadr "rgb" oqimidan olingan rasm bo'ladi, aniqlashlar nn natijalarini o'z ichiga oladi
    frame = None
    detections = []

    # Since the detections returned by nn have values from <0..1> range, they need to be multiplied by frame width/height to
    # receive the actual position of the bounding box on the image
    # Nn tomonidan qaytarilgan aniqlanishlar <0..1> diapazonidagi qiymatlarga ega bo'lgani uchun ularni ramka kengligi/balandligi bilan ko'paytirish kerak.
    # rasmdagi cheklovchi qutining haqiqiy holatini oling
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    # Main host-side application loop
    while True:
        # we try to fetch the data from nn/rgb queues. tryGet will return either the data packet or None if there isn't any
        # biz nn/rgb navbatlaridan ma'lumotlarni olishga harakat qilamiz. tryGet ma'lumotlar paketini yoki yo'q bo'lsa, Yo'qni qaytaradi
        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()

        if in_rgb is not None:
            # If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame
            # Agar RGB kamerasi paketi mavjud bo'lsa, biz getCvFrame yordamida OpenCV formatidagi freymni olinmoqda.
            frame = in_rgb.getCvFrame()
        if in_nn is not None:
            # when data from nn is received, we take the detections array that contains mobilenet-ssd results
            # nn dan ma'lumotlar olinganda, biz mobilenet-ssd natijalarini o'z ichiga olgan aniqlashlar qatorini olamiz
            detections = in_nn.detections

        if frame is not None:
            for detection in detections:
                # for each bounding box, we first normalize it to match the frame size
                # har bir cheklovchi quti uchun biz avval uni ramka o'lchamiga moslashtirish uchun normallashtiramiz
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                # and then draw a rectangle on the frame to show the actual result
                # va keyin haqiqiy natijani ko'rsatish uchun ramkaga to'rtburchak chizing
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            # After all the drawing is finished, we show the frame on the screen
            cv2.imshow("preview", frame)
        # at any time, you can press "q" and exit the main loop, therefore exiting the program itself
        # istalgan vaqtda siz "q" tugmasini bosishingiz va asosiy tsikldan chiqishingiz mumkin, shuning uchun dasturning o'zidan chiqishingiz mumkin
        if cv2.waitKey(1) == ord('q'):
            break