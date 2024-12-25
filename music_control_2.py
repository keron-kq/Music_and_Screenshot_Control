import pyvidu as vidu
import cv2 as cv
import numpy as np
import mediapipe as mp
import pyautogui
import time
import subprocess

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

# 追踪网易云音乐是否已经打开
music_opened = False

# 网易云音乐的路径
music_path = r"C:\WYCloud\install\CloudMusic\cloudmusic.exe"
double_click_detected = False

#用于计算手势方向
previous_index_finger_tip = None
index_finger_path = []
min_circle_points = 10  # 至少需要10个点形成圆圈
angle_threshold = 30  # 角度变化阈值，以度为单位

#检测手画圆
# def detect_circle_path(path, threshold):
#     if len(path) < 5:
#         return False
#
#     # 检测路径的点是否围绕一个中心点运动
#     x_vals, y_vals = zip(*path)
#     x_center = np.mean(x_vals)
#     y_center = np.mean(y_vals)
#     radius = np.sqrt((np.array(x_vals) - x_center) ** 2 + (np.array(y_vals) - y_center) ** 2)
#
#     return np.mean(radius) < threshold

def calculate_angle(p1, p2, p3):
    def vector(p1, p2):
        return np.array([p2[0] - p1[0], p2[1] - p1[1]])

    def angle_between(v1, v2):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return np.arccos(dot_product / (norm_v1 * norm_v2)) * 180.0 / np.pi

    v1 = vector(p1, p2)
    v2 = vector(p2, p3)
    return angle_between(v1, v2)

# 检测圆圈手势
def detect_circle_path(path, angle_threshold):
    if len(path) < min_circle_points:
        return False

    angles = []
    for i in range(1, len(path) - 1):
        angle = calculate_angle(path[i - 1], path[i], path[i + 1])
        angles.append(angle)

    mean_angle = np.mean(angles)
    return np.all(np.abs(np.array(angles) - mean_angle) < angle_threshold)

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
    action_threshold = 0.2 #动作阈值
    z_threshold = 0.15  # z轴阈值，检测快速接近和离开的动作
    circle_threshold = 50 #圆圈阈值
    z_values = []
    single_click_detected = False  # 用于跟踪是否已经检测到单次点击
    single_click_duration_threshold = 3


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
                        cv.circle(mat2, (int(corrected_x), int(corrected_y)), 5, (0, 255, 255), -1)  # 青色
                    else:
                        cv.circle(mat1_resized, (x, y), 5, (0, 255, 0), -1)  # 绿色
                        cv.circle(mat2, (int(corrected_x), int(corrected_y)), 5, (255, 255, 255), -1)  # 白色

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
                    cv.line(mat2, (int(corrected_start_x), int(corrected_start_y)), (int(corrected_end_x), int(corrected_end_y)), (255, 255, 255), 2)  # 使用白色

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
                            # print("Double click detected!")
                            double_click_detected = True
                            # 检查网易云音乐是否已经打开
                            if not music_opened:
                                # 打开网易云音乐软件
                                subprocess.Popen(music_path)
                                music_opened = True  # 更新状态为已打开
                                print("Double click detected!, Opened Cloud Music")
                            # 清除点击时间戳，防止多次执行
                            click_timestamps = []
                            single_click_detected = False  # 重置单击检测状态

                        # # 检测单击
                        # elif len(click_timestamps) == 1 and music_opened:
                        #     time.sleep(single_click_duration_threshold)
                        #     if not single_click_detected and len(click_timestamps) == 1:
                        #         print("Single click detected!")
                        #         # 模拟 Ctrl+P 键以控制音乐暂停/播放
                        #         pyautogui.hotkey('ctrl', 'p')
                        #         single_click_detected = True  # 更新状态为已检测到单次点击
                        #
                        #     # 清除点击时间戳，防止多次执行
                        #     click_timestamps = []
                else:
                    # 重置检测状态
                    single_click_detected = False

                # 检测食指指尖的移动方向
                if previous_index_finger_tip is not None:
                    index_finger_path.append((index_finger_tip.x, index_finger_tip.y))
                    if len(index_finger_path) > 20:  # 限制存储的路径点数量
                        index_finger_path.pop(0)
                    # 检测圆圈手势
                    # if detect_circle_path(index_finger_path, angle_threshold):
                    #     pyautogui.hotkey('ctrl', 'p')
                    #     print("Toggled Play/Pause")


                    dx = index_finger_tip.x - previous_index_finger_tip.x
                    dy = index_finger_tip.y - previous_index_finger_tip.y
                    dz = index_finger_tip.z - previous_index_finger_tip.z
                    # dx = x - previous_index_finger_tip[0]
                    # dy = y - previous_index_finger_tip[1]

                    # 左右移动
                    if abs(dx) > abs(dy):
                        if dx > 0.5 * action_threshold:
                            pyautogui.hotkey('ctrl', 'left')  # 下一首歌
                            print("Previous song")
                        elif dx < -0.5 * action_threshold:
                            pyautogui.hotkey('ctrl', 'right')  # 上一首歌
                            print("Next song")

                    else:
                        if dy > action_threshold:
                            pyautogui.hotkey('ctrl', 'down')
                            print("Decrease Volume")
                        elif dy < -action_threshold:
                            pyautogui.hotkey('ctrl', 'up')
                            print("Increase Volume")

                    #暂停或播放
                    if abs(dz) > action_threshold:
                        pyautogui.hotkey('ctrl','p')
                        print("Toggled Play/Pause")

                previous_index_finger_tip = index_finger_tip

        depth_colored = cv.applyColorMap(cv.convertScaleAbs(mat2, alpha=0.03), cv.COLORMAP_JET)
        cv.imshow("RGB Stream", mat1_resized)
        cv.imshow("ToF Stream",  depth_colored)
        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break

cv.destroyAllWindows()
