import pyvidu as vidu
import cv2 as cv
import numpy as np
import mediapipe as mp
import pyautogui
import time

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

# 定义视差校正参数（根据实际情况调整）
dx = -10  # x方向校正偏移
dy = 0   # y方向校正偏移

# 存储指尖轨迹的列表
finger_tip_trajectory_rgb = []
finger_tip_trajectory_tof = []

# 使用第一个数据流作为 RGB 流
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

    previous_index_finger_tip = None
    click_timestamps = []
    action_threshold = 0.1  # 动作阈值，检测移动的距离
    click_duration_threshold = 2  # 双击时间间隔阈值（秒）
    z_threshold = 0.25  # z轴阈值，检测快速接近和离开的动作
    z_values = []  # 用于平滑处理的z轴值列表

    screenshot_path = "screenshot.png"  # 截图保存路径

    while True:
        # 读取 RGB 数据流
        frame1 = rgb_stream.getPyMat()
        if not frame1:
            print("Failed to read RGB frames")
            continue

        # 读取 ToF 数据流
        frame2 = tof_stream.getPyMat()
        if not frame2:
            print("Failed to read ToF frames")
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
                    # 获取骨骼点的坐标
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    # 应用视差校正
                    corrected_x = x + dx
                    corrected_y = y + dy

                    # 在RGB图像和深度图中绘制骨骼点
                    if i == mp_hands.HandLandmark.INDEX_FINGER_TIP:
                        # 使用不同颜色绘制食指指尖
                        cv.circle(mat1_resized, ( x, y), 5, (0, 0, 255), -1)  # 红色
                        cv.circle(mat2, (corrected_x, corrected_y), 5, (0, 255, 255), -1)  # 青色
                    else:
                        cv.circle(mat1_resized, (x, y), 5, (0, 255, 0), -1)  # 绿色
                        cv.circle(mat2, (corrected_x, corrected_y), 5, (255, 255, 255), -1)  # 白色

                # 绘制骨骼连线
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

                # 获取食指指尖的坐标
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x = int(index_finger_tip.x * mat1.shape[1])
                y = int(index_finger_tip.y * mat1.shape[0])
                z = index_finger_tip.z

                # 记录指尖的轨迹
                finger_tip_trajectory_rgb.append((x, y))
                finger_tip_trajectory_tof.append((corrected_x, corrected_y))

                # 在RGB图像和深度图像中绘制食指指尖的运动轨迹
                for i in range(1, len(finger_tip_trajectory_rgb)):
                    cv.line(mat1_resized, finger_tip_trajectory_rgb[i - 1], finger_tip_trajectory_rgb[i], (0, 0, 255), 2)
                for i in range(1, len(finger_tip_trajectory_tof)):
                    cv.line(mat2, finger_tip_trajectory_tof[i - 1], finger_tip_trajectory_tof[i], (0, 255, 255), 2)

                # 检测食指指尖的移动方向
                if previous_index_finger_tip is not None:
                    dx = index_finger_tip.x - previous_index_finger_tip.x
                    dy = index_finger_tip.y - previous_index_finger_tip.y

                    # 左右移动
                    if dx > action_threshold:
                        pyautogui.press('right')  # 下一张图片
                        # print("Next image")
                    elif dx < -action_threshold:
                        pyautogui.press('left')  # 上一张图片
                        # print("Previous image")

                    # 上下移动
                    if dy > action_threshold:
                        pyautogui.press('pagedown')  # 文档下一页
                        # print("Next page")
                    elif dy < -action_threshold:
                        pyautogui.press('pageup')  # 文档上一页
                        # print("Previous page")

                previous_index_finger_tip = index_finger_tip

                # 调试打印z轴值
                print(f"Index finger z: {z}")

                # 将z轴值添加到列表中
                z_values.append(z)
                if len(z_values) > 5:  # 平滑处理，使用最近5个z轴值的平均值
                    z_values.pop(0)
                smoothed_z = np.mean(z_values)

                # 检测点击动作
                if len(click_timestamps) == 0 or (time.time() - click_timestamps[-1]) > 0.5:
                    if smoothed_z < -z_threshold:  # 检测快速接近
                        click_timestamps.append(time.time())

                        # 检查是否在2秒内连续点击两次
                        if len(click_timestamps) >= 2 and (click_timestamps[-1] - click_timestamps[-2]) < click_duration_threshold:
                            # pyautogui.screenshot(screenshot_path)
                            print("Screenshot taken")

                            # 显示截图
                            screenshot = cv.imread(screenshot_path)
                            if screenshot is not None:
                                cv.imshow("Screenshot", screenshot)
                            else:
                                print("Failed to load screenshot")

                            # 清空点击时间戳列表
                            click_timestamps = []

        # 将深度图转换为彩色图像
        depth_colored = cv.applyColorMap(cv.convertScaleAbs(mat2, alpha=0.03), cv.COLORMAP_JET)

        # 显示 RGB 图像
        cv.imshow("RGB Stream", mat1_resized)

        # 显示彩色深度图
        cv.imshow("ToF Stream", depth_colored)

        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break

cv.destroyAllWindows()
