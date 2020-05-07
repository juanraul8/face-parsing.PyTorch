#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from timeit import default_timer as timer

#device = 'cpu'
device = 'cuda:0'

def vis_parsing_maps(im, parsing_anno, size, stride, save_path='"test'):

    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.resize(vis_parsing_anno_color, size, interpolation=cv2.INTER_AREA)

    cv2.imwrite(save_path + '.png', vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    vis_parsing_anno = cv2.resize(vis_parsing_anno, size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(save_path + '_raw.png', vis_parsing_anno)

    #Skin image
    skin = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))
    index = np.where(vis_parsing_anno == 1)
    skin[index[0], index[1], :] = [255, 255, 255]

    skin = cv2.resize(skin, size, interpolation=cv2.INTER_AREA)

    skin_img = skin.astype(np.uint8)
    cv2.imwrite(save_path + '_skin.png', skin_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    skin = skin/255.0
    np.save(save_path + '_skin.npy', skin)

    return vis_im

def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

if __name__ == "__main__":

    start = timer()

    data_folder = "/mnt/raid/juan/StyleGANRelightingData/"

    n_classes = 19
    net = BiSeNet(n_classes=n_classes).to(device)
    net = BiSeNet(n_classes=n_classes)
    net.load_state_dict(torch.load("res/cp/79999_iter.pth", map_location=torch.device(device)))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    for root, dirs, files in os.walk(data_folder):

        print("Walking folders")

        dirs.sort()

        depth = root.count(os.sep) - data_folder.count(os.sep)

        # Debugging
        # print(root)
        # print (dirs)
        # print (files)
        # print (depth)

        if root == data_folder or depth > 0:
            continue

        for dir in dirs:
            folder = os.path.join(root, dir)

            print("Processing folder: {}".format(folder))

            labelPath = os.path.join(folder, 'label')

            if not os.path.exists(labelPath):
                os.makedirs(labelPath)

            with torch.no_grad():

                # Load image
                id = int(dir.split("_")[1])
                name = "face_{:02d}".format(id)

                file = os.path.join(folder, "{}.png".format(name))

                img = Image.open(file)
                H, W = img.size
                #image = img.resize((512, 512), Image.BILINEAR)
                image = img

                img = to_tensor(image)
                img = torch.unsqueeze(img, 0)
                img.to(device)

                out = net(img)[0]

                parsing = out.squeeze(0).cpu().numpy().argmax(0)

                # print(parsing)

                out_file = os.path.join(labelPath, name)

                vis_parsing_maps(image, parsing, stride=1, size=(H, W), save_path=out_file)

            #print("Test Done")
            #input()

        #print("Test Done")
        #input()

    end = timer()
    print("Face Labeling is Done: {:0.2f} (s)".format(end - start))