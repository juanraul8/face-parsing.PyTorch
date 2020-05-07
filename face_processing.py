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

    n_classes = 19
    net = BiSeNet(n_classes=n_classes).to('cpu')
    net.load_state_dict(torch.load("res/cp/79999_iter.pth", map_location=torch.device('cpu')))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    H, W = (1024, 1024)
    data_folder = "/mnt/raid/juan/relight_dataset/pablo_palafox/faces"

    files = [x for x in os.listdir(data_folder) if x.endswith(".png")]
    paths = [""] * len(files)

    # print (files)

    # Load files by index
    #for file in files:
    for i in range(0, 100):
        #tokens = file.split(".")

        #id = int(tokens[0].split("e")[1])
        #path = os.path.join(data_folder, file)

        id = 0
        path = "/home/juan/Face_Parsing/Face.png"

        print (path)

        with torch.no_grad():
            img = Image.open(path)
            image = img.resize((512, 512), Image.BILINEAR)

            #image = img
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img.to('cpu')

            out = net(img)[0]

            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            # print(parsing)

            out_file = "/home/juan/Face_Parsing/res/faces/semantic_face_{:04d}".format(id)

            vis_parsing_maps(image, parsing, stride=1, size = (H, W), save_path=out_file)

    end = timer()
    print("Preprocessing: {:0.2f} (s)".format(end - start))