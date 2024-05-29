import numpy as np

############################# 3d watershed ###############################
import SimpleITK as sitk
import os

def seg_frac_by_watershed(frac_list):
    """
    segment bone fractures from identification area
    # the result saved in './data/watershed_output' 
    # """
    frac_seg_list = []
    pp=0
    for frac_img in frac_list:
        pp += 1
        img = frac_img

        ## union and separation 
        seg = sitk.ConnectedComponent(img > 0)

        ## fill hole 
        filled = sitk.BinaryFillhole(seg != 0)
        d = sitk.SignedMaurerDistanceMap(filled, insideIsPositive=False, squaredDistance=True, useImageSpacing=True)

        ## adapted watershed
        classes = 10
        level = 1
        while classes > 3:
        # while classes > 2:
            watershed = sitk.MorphologicalWatershed(d, markWatershedLine=False, level=level, fullyConnected=True)
            watershed = sitk.Mask(watershed, sitk.Cast(seg, watershed.GetPixelID()))
            classes = len(np.unique(watershed))
            # print("classes", classes)
            if level <= 10:
                level += 1
            else:
                level += 10

        ## number of classes
        classes = len(np.unique(watershed))
        areas = []
        ws_array = sitk.GetArrayFromImage(watershed)

        ## area of each classes
        for i in range(1, classes + 1):
            area = int(np.sum(ws_array[ws_array == i]))
            # print(area)
            areas.append(area)

        # watershed = sitk.GetImageFromArray(ws_array)
        frac_seg_list.append(watershed)
        output_path = 'tempt/water_' + str(pp) + '.nii.gz'

        # # print("output_path", output_path)
        sitk.WriteImage(watershed, output_path)
    return frac_seg_list