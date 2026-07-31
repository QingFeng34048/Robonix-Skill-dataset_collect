'''
此脚本用于检测摄像头设备是否正确连接
'''


import cv2
import time

def check_camera():
    camera_id = 2
    print(f"正在尝试打开摄像头 /dev/video{camera_id} ...")
    
    # 尝试打开
    cap = cv2.VideoCapture(camera_id)
    
    # 设置分辨率 (奥比中光 Dabai DC1 推荐 640x480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 检查是否成功
    if not cap.isOpened():
        print(f"<span class=\"emoji emoji2716\"></span> 失败：无法打开摄像头 {camera_id}")
        print("常见原因：")
        print("1. 权限不足（尝试运行：sudo chmod 777 /dev/video2）")
        print("2. 设备号被占用")
        return

    print("成功：摄像头已打开！")
    print("按 'q' 键退出预览。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("<span class=\"emoji emoji2716\"></span> 错误：无法读取画面帧")
            break

        # 显示画面
        cv2.imshow('Piper Arm Camera (Orbbec)', frame)

        # 按 q 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    check_camera()