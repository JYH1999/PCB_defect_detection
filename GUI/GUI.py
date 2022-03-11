import tkinter
from threading import Thread
from tkinter.filedialog import askopenfilename
import numpy
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
import time

root = tkinter.Tk()
root.title('PCB缺陷检测')
root.geometry('860x640+400+100')
gpu_boost=False#是否选用GPU加速
yolo_config_path='yolov4-tiny-pcb.cfg'#YOLO配置文件
yolo_weights_path='yolov4-tiny-pcb_93.weights'#YOLO权重文件
yolo_label_path='pcb.names'
video_w=860  #长度
video_h=640  #宽度
dbg_mode=True#调试输出

local_detect=0#本地识别触发量
local_temp_path=""#本地图片目录缓存
camera_detect=0#摄像头识别触发量

# 用来显示视频画面的Label组件，自带双缓冲，不闪烁
lbVideo = tkinter.Label(root, bg='white')
lbVideo.pack(fill=tkinter.BOTH, expand=tkinter.YES)
# 创建菜单
mainMenu = tkinter.Menu(root)
subMenu = tkinter.Menu(tearoff=0)

if gpu_boost==True:
    print('从硬盘加载YOLO......',end="")
    net = cv2.dnn.readNetFromDarknet(yolo_config_path, yolo_weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    LABELS = open(yolo_label_path,encoding='UTF-8').read().strip().split("\n")
    nclass = len(LABELS)
    # 为每个类别的边界框随机匹配相应颜色
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(nclass, 3), dtype='uint8')
    print("Done!")
else:
    print('从硬盘加载YOLO......',end="")
    net = cv2.dnn.readNetFromDarknet(yolo_config_path, yolo_weights_path)
    LABELS = open(yolo_label_path,encoding='UTF-8').read().strip().split("\n")
    nclass = len(LABELS)
    # 为每个类别的边界框随机匹配相应颜色
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(nclass, 3), dtype='uint8')
    print("Done!")

def yolo_detect(pathIn='',
                pathOut=None,
                confidence_thre=0.05,
                nms_thre=0.3,
                jpg_quality=80,
                image_in='',
                result_output=False):

    # # 加载类别标签文件
    # LABELS = open(label_path,encoding='UTF-8').read().strip().split("\n")
    # nclass = len(LABELS)
    # # 为每个类别的边界框随机匹配相应颜色
    # np.random.seed(45)
    # COLORS = np.random.randint(0, 255, size=(nclass, 3), dtype='uint8')
    # 载入图片并获取其维度
    base_path = os.path.basename(pathIn)
    
    img=image_in
    #img = cv2.imread(pathIn)
    (H, W) = img.shape[:2]
    # 加载模型配置和权重文件
    # print('从硬盘加载YOLO......')
    # net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    # 获取YOLO输出层的名字
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # 将图片构建成一个blob，设置图片尺寸，然后执行一次
    # YOLO前馈网络计算，最终获取边界框和相应概率
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (1024, 1024), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    # 显示预测所花费时间
    print('YOLO模型花费 {:.2f} 秒来预测一张图片'.format(end - start))
    # 初始化边界框，置信度（概率）以及类别
    boxes = []
    confidences = []
    classIDs = []
    # 迭代每个输出层，总共三个
    for output in layerOutputs:
        # 迭代每个检测
        for detection in output:
            # 提取类别ID和置信度
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # 只保留置信度大于某值的边界框
            if confidence > confidence_thre:
                # 将边界框的坐标还原至与原图片相匹配，记住YOLO返回的是
                # 边界框的中心坐标以及边界框的宽度和高度
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # 计算边界框的左上角位置
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # 更新边界框，置信度（概率）以及类别
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    # 使用非极大值抑制方法抑制弱、重叠边界框
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thre, nms_thre)
    # 确保至少一个边界框
    if len(idxs) > 0:
        # 迭代每个边界框
        for i in idxs.flatten():
            # 提取边界框的坐标
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # 绘制边界框以及在左上角添加类别标签和置信度
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = '{}: {:.3f}'.format(LABELS[classIDs[i]], confidences[i])
            text.encode('utf-8')
            print(text)#输出种类
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x, y - text_h - baseline), (x + text_w, y), color, -1)
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    # 输出结果图片
    if result_output==True:
        if pathOut is None:
            cv2.imwrite('with_box_' + base_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
        else:
            cv2.imwrite(pathOut, img, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
    return img

def image_detect(image_input='test.jpg',image_obj='',dbg_output=False,label_show=True,fig_save=False):#图片识别函数
    if dbg_output==True:
        print("Function:image_detect")
    detect_confidence_thre=0.1 #置信率阈值
    output_path="./output/"+os.path.basename(image_input)
    image_output=output_path
    img_return=yolo_detect(pathIn=image_input,
                pathOut=image_output,
                confidence_thre= detect_confidence_thre,
                nms_thre=0.3,
                jpg_quality=100,
                result_output=fig_save,
                image_in=image_obj)
    if label_show==True:
        image_temp = cv2.cvtColor(img_return, cv2.COLOR_BGR2RGB)
        img_output=Image.fromarray(image_temp)
        pw=img_output.width
        ph=img_output.height
        output_ratio = min(video_w / pw, video_h / ph)
        size = (int(pw * output_ratio), int(ph * output_ratio))
        if dbg_output==True:
            print("label_show==True，刷新页面")
            print('图片原始宽度'+str(pw))
            print('图片原始高度'+str(ph))
            print('图片缩放比例'+str(output_ratio))
            print('图片缩放尺寸'+str(size[0])+'*'+str(size[1]))
        frame_output = Image.fromarray(numpy.array(img_output)).resize(size)
        frame_output = ImageTk.PhotoImage(frame_output)
        lbVideo['image'] = frame_output
        lbVideo.image = frame_output
        lbVideo.update()

def open_file():

    fn = askopenfilename(title='打开图片文件',
                         filetypes=[('图片', '*.jpg *.jpeg')])
    return fn

def gstreamer_pipeline(#使用gstreamer生成摄像头参数
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
 
def camera_init(dbg_output=False):#初始化相机
    if dbg_output==True:
        print("Function:camera_init")
    try:
        cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
        if dbg_output==True:
            if cap.isOpened():
                print("Camera open successfully")
            else:
                print("Warning:Camera initialized, but not at an open mode!")
    except:
        print("Failed to open camera")
    return cap
    
def camera_capture(cap,dbg_output=False):#相机拍照
    if dbg_output==True:
        print("Function:camera_capture")
    if cap.isOpened():
        try:
            img=''
            #flag,img=cap.read()
            for i in range(10):
                flag, img = cap.read()
            #cv2.imwrite(img_path,img)#多写一次解除文件锁
            if dbg_output==True:
                print("Camera captured & photo saved")
                return img
        except:
            print("Failed to capture, retry opening camera")
            cap=camera_init(dbg_output=dbg_output)
    else:
        if dbg_output==True:
            print("Camera is not opened, try to open camera")
            cap=camera_init(dbg_output=dbg_output)

def open_and_recognize():
    global local_detect,local_temp_path
    fp=open_file()
    local_temp_path=fp
    local_detect=1

def cam_recognize():
    global camera_detect
    if camera_detect==1:
        camera_detect=0
    else:
        camera_detect=1

def detect_threads():#检测线程函数
    global dbg_mode,local_detect,camera_detect
    camera=camera_init(dbg_output=dbg_mode)#初始化摄像头
    while True:
        if local_detect==1:
            img_obj = cv2.imdecode(np.fromfile(local_temp_path, dtype=np.uint8), -1)
            image_detect(image_input=local_temp_path,dbg_output=dbg_mode,label_show=True,image_obj=img_obj,fig_save=True)
            local_detect=0
        elif camera_detect==1:
            image_captured=camera_capture(camera,dbg_output=dbg_mode)
            image_captured=''
            image_captured=camera_capture(camera,dbg_output=dbg_mode)#拍第二次，解决获得前一次照片的bug
            image_detect(image_input="",dbg_output=dbg_mode,label_show=True,image_obj=image_captured,fig_save=False)
        time.sleep(0.1)

# 添加菜单项，设置命令
subMenu.add_command(label='本地',
                    command=open_and_recognize)
subMenu.add_command(label='相机',
                    command=cam_recognize)
# 把子菜单挂到主菜单上
mainMenu.add_cascade(label='识别',
                     menu=subMenu)
# 把主菜单放置到窗口上
root['menu'] = mainMenu

detect_threading=Thread(target=detect_threads,name='detect_threading')
detect_threading.daemon =True
detect_threading.start()
print("Start detect thread")

root.mainloop()
