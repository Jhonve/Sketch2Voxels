import os
import glob
import re
import scipy.io as sio
import cv2
import numpy as np

def getFileSetList(dir):
    return glob.glob(os.path.join(dir, "*"))

def getDataInfo(list, filename):
    dir = [l for l in list if re.search(filename, os.path.basename(l))]
    return dir[0]

def readGray(file):
    # read mask and depth
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32)
    return img

def readImg(file):
    # read 2d orientation field
    img = cv2.imread(file, cv2.IMREAD_COLOR)
    img = img.astype(np.float32)
    return img

def loadImgInput(list):
    sketch_dir = getDataInfo(list, "sketch.jpg")
    sketch_img = readGray(sketch_dir)
    input_data = np.expand_dims(sketch_img, axis=2)

    return input_data * (1. / 255.)

def loadGTOutput(list):
    normal_dir = getDataInfo(list, "normal.jpg")
    normal_img = readImg(normal_dir)
    
    depth_dir = getDataInfo(list, "depth.jpg")
    depth_img = readGray(depth_dir)
    depth_img = np.expand_dims(depth_img, axis=2)

    gt_img = np.concatenate((normal_img, depth_img), axis=2)

    output_data = gt_img * (1. / 255.)
    return output_data

def getTrainBatch(list, offset, batch_size):
    batch_list = list[offset : offset + batch_size]
    img_data = []
    gt_value = []
    for f in batch_list:
        file_list = getFileSetList(f)
        input_img = loadImgInput(file_list)
        output_gt = loadGTOutput(file_list)
        img_data.append(input_img)
        gt_value.append(output_gt)

    img_data = np.array(img_data)
    img_data = np.resize(img_data, [batch_size, 256, 256, 1])
    gt_value = np.array(gt_value)
    gt_value = np.resize(gt_value, [batch_size, 256, 256, 4])
    return img_data, gt_value

def draw2DOri(value, batch_size):
    if(batch_size > 1):
        img = value[0]
        for i in range(1, batch_size):
            temp = value[i]
            img = np.concatenate((img, temp), axis = 1)
        normal_img = img[:, :, :3]
        # normal_img = normal_img[..., ::-1]
        depth_img = img[:, :, 3]
        return normal_img * 255, depth_img * 255
    else:
        value = np.array(value)
        print(value.shape)
        img = value[0].reshape((256, 256, 4))
        normal_img = img[:, :, :3]
        # normal_img = normal_img[..., ::-1]
        depth_img = img[:, :, 3]
        return normal_img * 255, depth_img * 255
