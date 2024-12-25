import pyvidu as vidu
import cv2 as cv
import numpy as np
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

intrinsic = vidu.intrinsics()
extrinsic = vidu.extrinsics()

device = vidu.PDdevice()
if not device.init():
    print("device init failed")
    exit(-1)

print("device init succeed  ", device.getSerialsNumber())
stream_num = device.getStreamNum()
print("stream_num = {}\n".format(stream_num))

dx = -10  # x方向校正偏移
dy = 0   # y方向校正偏移

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

    tof_stream.set("Distance", 5.0)
    tof_stream.set("StreamFps", 30)
    tof_stream.set("AutoExposure", True)
    tof_stream.set("Exposure", 420)
    tof_stream.set("Gain", 1.0)
    tof_stream.set("Threshold", 0)
    tof_stream.set("DepthFlyingPixelRemoval", 0)
    tof_stream.set("DepthSmoothStrength", 0)

    click_timestamps = []
    click_duration_threshold = 2  # 两次点击的时间间隔阈值（秒）
    z_threshold = 0.25  # z轴阈值，检测快速接近和离开的动作

    z_values = []

    screenshot_path = "screenshot.png"

    while True:
        # 读取 RGB 数据流
        frame1 = rgb_stream.getPyMat()
        if not frame1:
            # print("Failed to read RGB frames")
            continue

        # 读取 ToF 数据流
        frame2 = tof_stream.getPyMat()
        if not frame2:
            # print("Failed to read ToF frames")
            continue

        # 取第一帧作为显示内容
        mat1 = frame1[0]
        mat2 = frame2[0]

        # 调整 RGB 图像大小以匹配 ToF 图像的高度
        height, width = mat2.shape
        mat1_resized = cv.resize(mat1, (width, height))

        # 使用 MediaPipe 检测手掌骨骼
        rgb_image = cv.cvtColor(mat1_resized, cv.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i, landmark in enumerate(hand_landmarks.landmark):
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    corrected_x = x + dx
                    corrected_y = y + dy

                    if i == mp_hands.HandLandmark.INDEX_FINGER_TIP:
                        cv.circle(mat1_resized, (x, y), 5, (0, 0, 255), -1)  # 红色
                        cv.circle(mat2, (corrected_x, corrected_y), 5, (0, 255, 255), -1)  # 青色
                    else:
                        cv.circle(mat1_resized, (x, y), 5, (0, 255, 0), -1)  # 绿色
                        cv.circle(mat2, (corrected_x, corrected_y), 5, (255, 255, 255), -1)  # 白色

                for connection in mp_hands.HAND_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    start_x = int(hand_landmarks.landmark[start_idx].x * width)
                    start_y = int(hand_landmarks.landmark[start_idx].y * height)
                    end_x = int(hand_landmarks.landmark[end_idx].x * width)
                    end_y = int(hand_landmarks.landmark[end_idx].y * height)

                    corrected_start_x = start_x + dx
                    corrected_start_y = start_y + dy
                    corrected_end_x = end_x + dx
                    corrected_end_y = end_y + dy

                    cv.line(mat1_resized, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                    cv.line(mat2, (corrected_start_x, corrected_start_y), (corrected_end_x, corrected_end_y), (255, 255, 255), 2)  # 使用白色

                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x = int(index_finger_tip.x * mat1.shape[1])
                y = int(index_finger_tip.y * mat1.shape[0])
                z = index_finger_tip.z



                # print(f"Index finger z: {z}")

                z_values.append(z)
                if len(z_values) > 5:  # 平滑处理，使用最近5个z轴值的平均值
                    z_values.pop(0)
                smoothed_z = np.mean(z_values)

                if len(click_timestamps) == 0 or (time.time() - click_timestamps[-1]) > 0.5:
                    if smoothed_z < -z_threshold:  # 检测快速接近
                        click_timestamps.append(time.time())

                        if len(click_timestamps) >= 2 and (click_timestamps[-1] - click_timestamps[-2]) < click_duration_threshold:
                            pyautogui.screenshot(screenshot_path)
                            print("Screenshot taken")

                            screenshot = cv.imread(screenshot_path)
                            if screenshot is not None:
                                cv.imshow("Screenshot", screenshot)
                            else:
                                print("Failed to load screenshot")

                            click_timestamps = []

        depth_colored = cv.applyColorMap(cv.convertScaleAbs(mat2, alpha=0.03), cv.COLORMAP_JET)
        cv.imshow("RGB Stream", mat1_resized)
        cv.imshow("ToF Stream",  depth_colored)
        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break

cv.destroyAllWindows()
