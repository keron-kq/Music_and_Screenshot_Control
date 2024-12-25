import pyvidu as vidu
import cv2 as cv
import numpy as np
import mediapipe as mp
import pyautogui

# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# 初始化 TOF 相机设备
intrinsic = vidu.intrinsics()
extrinsic = vidu.extrinsics()

device = vidu.PDdevice()
if not device.init():
    print("device init failed")
    exit(-1)

print("device init succeed  ", device.getSerialsNumber())
stream_num = device.getStreamNum()
print("stream_num = {}\n".format(stream_num))


with vidu.PDstream(device, 0) as rgb_stream, vidu.PDstream(device, 1) as tof_stream:

    suc_1 = rgb_stream.init()
    print(suc_1)
    stream_name_rgb = rgb_stream.getStreamName()
    print(f"Stream name: {stream_name_rgb}, init success: {suc_1}")

    suc_2 = tof_stream.init()
    print(suc_2)
    stream_name_tof = tof_stream.getStreamName()
    print(f"Stream name: {stream_name_tof}, init success: {suc_2}")

    rgb_stream.set("AutoExposure", False)
    rgb_stream.set("Exposure", 29984)
    rgb_stream.set("Gain", 24)
    rgb_stream.set("StreamFps", 30)
    # rgb_stream.set("DistortRomove", True)

    tof_stream.set("Distance", 5.0)
    tof_stream.set("StreamFps", 30)
    tof_stream.set("AutoExposure", True)
    tof_stream.set("Exposure", 420)
    tof_stream.set("Gain", 1.0)
    tof_stream.set("Threshold", 400)
    tof_stream.set("DepthFlyingPixelRemoval", 0)
    tof_stream.set("DepthSmoothStrength", 0)
    # tof_stream.set("DistortRomove", True)

    while True:
        frame1 = rgb_stream.getPyMat()
        if not frame1:
            # print("Failed to read RGB frames")
            continue

        frame2 = tof_stream.getPyMat()
        if not frame2:
            # print("Failed to read ToF frames")
            continue

        mat1 = frame1[0]
        mat2 = frame2[0]

        height, width = mat2.shape
        mat1_resized = cv.resize(mat1, (width, height))
        # print(mat1_resized.shape)

        rgb_image = cv.cvtColor(mat1_resized, cv.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i, landmark in enumerate(hand_landmarks.landmark):
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv.circle(mat1_resized, (x, y), 5, (0, 255, 0), -1)
                    cv.circle(mat2, (x, y), 5, (255, 255, 255), -1)


                for connection in mp_hands.HAND_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    start_x = int(hand_landmarks.landmark[start_idx].x * width)
                    start_y = int(hand_landmarks.landmark[start_idx].y * height)
                    end_x = int(hand_landmarks.landmark[end_idx].x * width)
                    end_y = int(hand_landmarks.landmark[end_idx].y * height)
                    cv.line(mat1_resized, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                    cv.line(mat2, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)

        depth_colored = cv.applyColorMap(cv.convertScaleAbs(mat2, alpha=0.03), cv.COLORMAP_TURBO)

        cv.imshow("RGB Stream", mat1_resized)
        cv.imshow("ToF Stream", depth_colored)

        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break

cv.destroyAllWindows()
