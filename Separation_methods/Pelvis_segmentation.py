import os
import torch
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from PIL import Image
from networks.pelvis_segmentation.unet import Unet

random_seed = 123
torch.manual_seed(random_seed)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda', 1)
# print(device)
input_shape = [512, 512]
def build_model(conf):
    # UNet
    ## backbone choose vgg or resnet50
    backbone = "vgg"
    # backbone = "resnet50"
    model = Unet(num_classes=conf['num_classes'], backbone=backbone)
    if conf['Muti_GPU']:
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(conf['prefix'], conf['weights']), map_location=device))
    model.to(device)
    model.eval()
    print("-------------Loading model done---------------")
    return model

def ImageResample_size(sitk_image, new_size = [512, 512, 166], is_label = False):
    '''
    sitk_image:
    new_spacing: x,y,z
    is_label: if True, using Interpolator `sitk.sitkNearestNeighbor`
    '''
    size = np.array(sitk_image.GetSize())
    spacing = np.array(sitk_image.GetSpacing())
    new_size = np.array(new_size)
    #new_spacing = size * spacing / new_size
    new_spacing_refine = size * spacing / new_size
    new_spacing_refine = [float(s) for s in new_spacing_refine]
    new_size = [int(s) for s in new_size]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing(new_spacing_refine)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        #resample.SetInterpolator(sitk.sitkBSpline)
        resample.SetInterpolator(sitk.sitkLinear)

    newimage = resample.Execute(sitk_image)
    return newimage

def cvtColor(image):
    image = Image.fromarray(image)
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a

def preprocess_data(image, input_shape):
    image = cvtColor(image)
    # get the image size and target size
    iw, ih = image.size
    h, w = input_shape

    dx = int(rand(0, w - iw))
    dy = int(rand(0, h - ih))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    image_data = np.array(image, np.uint8)
    return image_data

def pelvis_segmentation(input_image):
    d2_coarse = {
        'num_classes':3 + 1,
        'prefix': r'model',
        'weights': 'pelvis_segmentaiton/Final_model.pth',
        'Muti_GPU': False
    }
    model = build_model(d2_coarse)
    pred_images = []

    
    input_image_resample = ImageResample_size(input_image, new_size=[512, 512, input_image.GetSize()[2]], is_label=False)

    input_data = sitk.GetArrayFromImage(input_image_resample)

    slices, height, width = np.shape(input_data)[0], np.shape(input_data)[1], np.shape(input_data)[2]

    for i in tqdm(range(slices)):
        # get one slice
        slice_image = input_data[i, :, :]
        slice_image = preprocess_data(slice_image, (input_shape[1],input_shape[0]))

        ## predict
        image_data = slice_image / 255.0
        # image_data = slice_image
        image_data = np.expand_dims(np.transpose(np.array(image_data), (2, 0, 1)), 0)
        image_data = image_data.astype(np.float)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images = images.to(device)
            images = images.float()
            pred = model(images)[0]

            pred = F.softmax(pred.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
            pred_image = pred.astype(np.float)

            pred_images.append(pred_image)

    output_data = np.array(pred_images).astype('int32')
    output_image = sitk.GetImageFromArray(output_data)

    # resample
    output_image = ImageResample_size(output_image, new_size=[input_image.GetSize()[0], input_image.GetSize()[1], input_image.GetSize()[2]], is_label=True)
    output_image.CopyInformation(input_image)

    # sitk.WriteImage(output_image, "output_image.nii.gz")
    return output_image



