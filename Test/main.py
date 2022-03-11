import detect
import os
def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname

picture_path_list=list(findAllFile("./test/"))
for pathIn in picture_path_list:
    print(pathIn)
    pathOut = './result/'+'output_'+pathIn.lstrip("./test/")
    try:
        detect.yolo_detect(pathIn,pathOut)
    except:
        continue
#pathIn = 'img_11289.jpg'
#pathOut = 'output.jpg'

# 调用
#detect.yolo_detect(pathIn,pathOut)