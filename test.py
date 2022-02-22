"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np

from spade.model import Pix2PixModel
from spade.dataset import get_transform
from torchvision.transforms import ToPILImage
from PIL import Image
import cv2


def evaluate(labelmap):
    opt = {
        'label_nc': 182, # num classes in coco model
        'crop_size': 512,
        'load_size': 512,
        'aspect_ratio': 1.0,
        'isTrain': False,
        'checkpoints_dir': 'flask/pretrained/',
        'which_epoch': 'latest',
        'use_gpu': False
    }
    model = Pix2PixModel(opt)
    model.eval()

    image = Image.fromarray(np.array(labelmap).astype(np.uint8))

    transform_label = get_transform(opt, method=Image.NEAREST, normalize=False)
    # transforms.ToTensor in transform_label rescales image from [0,255] to [0.0,1.0]
    # lets rescale it back to [0,255] to match our label ids
    label_tensor = transform_label(image) * 255.0
    label_tensor[label_tensor == 255] = opt['label_nc'] # 'unknown' is opt.label_nc
    print("label_tensor:", label_tensor.shape)

    # not using encoder, so creating a blank image...
    transform_image = get_transform(opt)
    image_tensor = transform_image(Image.new('RGB', (500, 500)))

    data = {
        'label': label_tensor.unsqueeze(0),
        'instance': label_tensor.unsqueeze(0),
        'image': image_tensor.unsqueeze(0)
    }
    generated = model(data, mode='inference')
    print("generated_image:", generated.shape)

    return generated

def to_image(generated):
    to_img = ToPILImage()
    normalized_img = ((generated.reshape([3, 512, 512]) + 1) / 2.0) * 255.0
    return to_img(normalized_img.byte().cpu())

def img_to_oilpaint(image):
    net = cv2.dnn.readNetFromTorch('flask/models/eccv16/starry_night.t7')

    img =np.array(image)

    # 전처리 
    h, w, c = img.shape

    img = cv2.resize(img, dsize=(500, int(h/w * 500)))

    img = img[162:513, 185:428]

    MEAN_VALUE = [103.933, 116.779, 123.680]
    blob = cv2.dnn.blobFromImage(img, mean=MEAN_VALUE)

    # 후처리 
    net.setInput(blob)
    output = net.forward()

    output = output.squeeze().transpose((1,2,0))
    output += MEAN_VALUE

    output = np.clip(output, 0, 255)
    output = output.astype('uint8')

    output = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    return output