import cv2
import matplotlib.pyplot as plt
import numpy as np

# 读取图片
img = cv2.imread(r"D:\picture\xhb4.jpg", 1)    # 读取彩色图片
def coordinate(img):
    width=img.shape[0]
    height=img.shape[1]

    def apply_closing_application1(img,size):   #闭运算
    # 设置核
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))
    # 闭运算
        filter_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        return filter_img
    #闭运算
    def apply_closing_application2(img,size):
    # 设置核
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))
    # 闭运算
        filter_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        return filter_img
    #开运算
    def apply_opening_operation(image, kernel_size=(3, 3)):
        # 创建一个结构元素（核）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        # 应用腐蚀操作
        eroded_image = cv2.erode(image, kernel)
        # 应用膨胀操作，得到开运算结果
        opened_image = cv2.dilate(eroded_image, kernel)
        return opened_image
    #高斯滤波
    def apply_gaussian_blur(image, size, sigmaX,sigmaY):
        blurred_image = cv2.GaussianBlur(image, (size,size), sigmaX, sigmaY)
        return blurred_image
    # RGB到HSI的变换
    def rgb2hsi(image):
        b, g, r = cv2.split(image)                    # 读取通道
        r = r / 255.0                                # 归一化
        g = g / 255.0
        b = b / 255.0
        eps = 1e-6                                   # 防止除零

        img_i = (r + g + b) / 3                      # I分量

        img_h = np.zeros(r.shape, dtype=np.float32)
        img_s = np.zeros(r.shape, dtype=np.float32)
        min_rgb = np.zeros(r.shape, dtype=np.float32)
        # 获取RGB中最小值
        min_rgb = np.where((r <= g) & (r <= b), r, min_rgb)
        min_rgb = np.where((g <= r) & (g <= b), g, min_rgb)
        min_rgb = np.where((b <= g) & (b <= r), b, min_rgb)
        img_s = 1 - 3*min_rgb/(r+g+b+eps)                                            # S分量

        num = ((r-g) + (r-b))/2
        den = np.sqrt((r-g)**2 + (r-b)*(g-b))
        theta = np.arccos(num/(den+eps))
        img_h = np.where((b-g) > 0, 2*np.pi - theta, theta)                           # H分量
        img_h = np.where(img_s == 0, 0, img_h)

        img_h = img_h/(2*np.pi)                                                       # 归一化
        temp_s = img_s - np.min(img_s)
        temp_i = img_i - np.min(img_i)
        img_s = temp_s/np.max(temp_s)
        img_i = temp_i/np.max(temp_i)

        image_hsi = cv2.merge([img_h, img_s, img_i])
        return img_h, img_s, img_i, image_hsi
    # 边缘检测
    def Edge_detection(image):
        image1 = cv2.Canny(image, 128, 256)
        return image1
    # 平均差量
    def Mean_difference(img):
        width=img.shape[0]
        height=img.shape[1]
        h, s, i, hsi = rgb2hsi(img)
        total_s = np.sum(s)
        S = total_s / (width * height)
        total_i = np.sum(i)
        I = total_i / (width * height)
        l = 0
        D = []
        while l < width:
            r = 0
            line = []
            while r < height:
                line.append(0)
                r = r + 1
            D.append(line)
            l = l + 1

        for u in range(2,width-2):
            for v in range(2,height-2):
                D[u][v]=0
                for k in range(-2,3):
                    for j in range(-2,3):
                        D[u][v]+=(s[u+k][v+j]-S+i[u+k][v+j]-I)
                D[u][v]=D[u][v]/25
        for u in range(width):
            for v in range(height):
                if (D[u][v]<-0.43):
                    D[u][v]=0
                else:
                    D[u][v]=255
        return D
    #重建图像
    def Reconstruction_image(width,height,f):
        image = np.zeros((width, height), np.uint8)
        image[:, :] = f
        return image;
    #突发奇想的绝妙组合滤波
    def Combined_filtering(img,num):
        img=apply_gaussian_blur(img,5,0,0)
        for i in range(num):
            img2=Edge_detection(img)
            img2=cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
            img=cv2.subtract(img,img2)
            img=apply_closing_application1(img,3)
        return img
    # 弹孔坐标
    def Bullet_hole_coordinates(img):
        L=[]
        lower = np.array(0)
        upper = np.array(50)
        mask = cv2.inRange(img, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(len(contours))
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                L.append([cX,cY])
        return L

    img=Combined_filtering(img,3) # 组合滤波

    img=Mean_difference(img) # 计算平均差量矩阵

    img=Reconstruction_image(width,height,img) # 重建图像

    img=apply_closing_application2(img,3) # 闭运算

    L=Bullet_hole_coordinates(img)# 保存弹孔坐标

    cv2.imshow('Generated Image', img)
    cv2.waitKey(0)
    return L
L=coordinate(img)
print(L)













