import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm

def rebuid_from_seg(frac_seg_list, predict_label, index_array):
    """
    输入骨折区域obb顶点坐标
    读取分割结果和原始label
    进行复原
    保存复原结果label到 'data/rebuild_output'
    """
    ## 存储左右两侧碎块类别
    right_list = []
    left_list = []

    ## 分别存储每个骨折区域的坐标
    frac_xmins = []
    frac_xmaxs = []
    frac_ymins = []
    frac_ymaxs = []
    frac_zmins = []
    frac_zmaxs = []

    ## 分别存储联通区域生长之后的只保留类别1和类别2的区域
    """
        frac_region_1 骨折区域只保留类别1的区域
        frac_region_2 骨折区域只保留类别2的区域
    """
    frac_region_1 = []
    frac_region_2 = []
    frac_region_3 = []

    num_frac_use = 0

    # label_path = label_root + '/' + 'label' + image[5:]

    label = predict_label
    label_array_part = []
    label_array_part_t = [] # 后面用来约束输出范围

    """
    按解剖结构划分
    i从1到3
    label_array_part[1] - 骶骨
    label_array_part[2] - 右髂骨
    label_array_part[3] - 左髂骨
    """

    label_array_1 = sitk.GetArrayFromImage(label)
    label_array_1[label_array_1 != 1] = 0
    label_array_part.append(label_array_1)

    label_array_1_t = label_array_1.copy()  # 后面用来约束输出范围
    label_array_part_t.append(label_array_1_t)

    label_array_2 = sitk.GetArrayFromImage(label)
    label_array_2[label_array_2 != 2] = 0
    label_array_part.append(label_array_2)

    label_array_2_t = label_array_2.copy()  # 后面用来约束输出范围
    label_array_2_t[label_array_2_t == 2] = 1
    label_array_part_t.append(label_array_2_t)

    label_array_3 = sitk.GetArrayFromImage(label)
    label_array_3[label_array_3 != 3] = 0
    label_array_part.append(label_array_3)

    label_array_3_t = label_array_3.copy()  # 后面用来约束输出范围
    label_array_3_t[label_array_3_t == 3] = 1
    label_array_part_t.append(label_array_3_t)

    """
        与分水岭设置的保留类别的超参数对应，默认为3类
        label_array1 为label中将每个检测出的骨折区域只保留类别1
        label_array2 为label中将每个检测出的骨折区域只保留类别2
        label_array3 为label中将每个检测出的骨折区域只保留类别3
    """

    for frac_seg in frac_seg_list:
        frac_seg_array_1 = sitk.GetArrayFromImage(frac_seg)
        frac_seg_array_2 = sitk.GetArrayFromImage(frac_seg)
        frac_seg_array_3 = sitk.GetArrayFromImage(frac_seg)

        num_frac_use = num_frac_use + 1
        """
           每一个断裂处根据分割结果重新划分为最多三块（多数为两块）
           第一块label_seg_array_1 只有类别1
           第二块label_seg_array_2 只有类别2
           第三块label_seg_array_3 只有类别3
       """

        ################# 先将每个断裂块变成两类，以便处理 ####################
        # frac_seg_array_1[frac_seg_array_1 > 2] = 2
        # frac_seg_array_2[frac_seg_array_2 > 2] = 2

        frac_seg_array_1[frac_seg_array_1 != 1] = 0
        frac_seg_array_2[frac_seg_array_2 != 2] = 0
        frac_seg_array_3[frac_seg_array_3 != 3] = 0

        # # ############ 为了使得骨折块和原始三块区分开来，因此骨折从8开始计数
        frac_seg_array_1[frac_seg_array_1 == 1] = 8
        frac_seg_array_2[frac_seg_array_2 == 2] = 9
        frac_seg_array_3[frac_seg_array_3 == 3] = 10

        frac_region_1.append(frac_seg_array_1)
        frac_region_2.append(frac_seg_array_2)
        frac_region_3.append(frac_seg_array_3)

        for i in range(index_array.shape[0]):
            xmin = index_array[i][0]
            xmax = index_array[i][1]
            ymin = index_array[i][2]
            ymax = index_array[i][3]
            zmin = index_array[i][4]
            zmax = index_array[i][5]

            frac_xmins.append(xmin)
            frac_xmaxs.append(xmax)
            frac_ymins.append(ymin)
            frac_ymaxs.append(ymax)
            frac_zmins.append(zmin)
            frac_zmaxs.append(zmax)

            for j in range(1, 3):
                label_array_part[j][zmin:zmax, ymin:ymax, xmin:xmax] = 0

    """
    将两块取出的断裂区域进行相加，即可得到完整断裂区域，完成断骨分割
    """
    ############## 将骨折区域去掉，进行区域生长，作为基本块 #############
    ##### 分别存储三个结构块中，去除骨折区域，然后区域生长后的每一个区域
    label_part_add_frac_region_arrays = []

    ########## 暂时先不考虑骶骨 ###############
    # for i in range(3):
    label_part_add_frac_region_arrays.append(label_array_part[0])
    for i in range(1, 3):
        label_part = sitk.GetImageFromArray(label_array_part[i])

        """
            对去除骨折区域之后的结果做腐蚀运算，断开相连的部分。
        """
        # filled = sitk.BinaryErodeImageFilter()
        # filled.SetForegroundValue(i + 1)
        # filled.SetKernelType(sitk.sitkBall)
        # filled.SetKernelRadius(2)
        # label_part = filled.Execute(label_part)
        # label_array_part[i] = sitk.GetArrayFromImage(label_part)

        # 连通域划分
        cc_filter = sitk.ConnectedComponentImageFilter()
        cc_filter.SetFullyConnected(True)
        output_mask_tem = cc_filter.Execute(label_part)
        num_connected_label_3 = cc_filter.GetObjectCount()

        # 获取各标签形状属性
        lss_filter = sitk.LabelShapeStatisticsImageFilter()
        lss_filter.Execute(output_mask_tem)  # 分析连通域

        """
            对每个类别进行膨胀。由于该步骤生成的label在后面要用原来的label做mask，因此kernel radius要比原来的label大。
        """
        # for j in range(0, num_connected_label_3):
        #     output_mask_tem = sitk.DilateObjectMorphology(output_mask_tem, kernelRadius=(3, 3, 3), kernelType=sitk.sitkBall, objectValue=j + 1)
        #     # output_mask_tem = sitk.DilateObjectMorphology(output_mask_tem, kernelRadius=(4, 4, 4), kernelType=sitk.sitkBall, objectValue=j + 1)

        areas = []
        for j in range(1, num_connected_label_3 + 1):
            area = lss_filter.GetNumberOfPixels(j)
            areas.append(area)

        # 取出取出骨折区域，然后区域划分后的区域
        label_part_region_array = sitk.GetArrayFromImage(output_mask_tem)

        ### 进行排序，按面积从大到小进行排序编号
        areas = np.array(areas)
        for j in range(0, num_connected_label_3 // 2):
            before_class = np.argsort(-areas)[j] + 1
            current_class = j + 1

            label_part_region_array[label_part_region_array == before_class] = 20
            label_part_region_array[label_part_region_array == current_class] = before_class
            label_part_region_array[label_part_region_array == 20] = current_class

        ### 对于一些体积很小的碎块，统一到最大块里面
        for k in range(1, num_connected_label_3 + 1):
            # if np.sum(label_part_region_array[label_part_region_array == k]) < 2000:
            if np.sum(label_part_region_array[label_part_region_array == k]) < 3000:
                label_part_region_array[label_part_region_array == k] = 1

        # output_mask_tem = sitk.GetImageFromArray(label_part_region_array)
        # write_label_rebuild_3 = 'tempt/' + '1111_3.nii.gz'
        # sitk.WriteImage(output_mask_tem, write_label_rebuild_3)

        """
        每一次循环将类别处理后的骨折区域放进去时，会出现类别互相干扰，导致后面的骨折区域类别不对
        因此将类别处理好的骨折区域全部存放在fracture_region
        然后在用循环一起放进去
        """
        fracture_region = []

        ############## 分别判断骨折区域的一类与哪一个孤岛块相连，设置为统一类别 #############
        for j in range(num_frac_use):
            # print("frac_zmaxs[j]", frac_zmaxs[j])
            # print("label_part_region_array.shape[0]", label_part_region_array.shape[0])
            # print("frac_ymaxs[j]", frac_ymaxs[j])
            # print("label_part_region_array.shape[1]", label_part_region_array.shape[1])
            # print("frac_xmaxs[j]", frac_xmaxs[j])
            # print("label_part_region_array.shape[2]", label_part_region_array.shape[2])
            if int(frac_zmaxs[j]) >= label_part_region_array.shape[0]:
                frac_zmaxs[j] = label_part_region_array.shape[0]
            if int(frac_ymaxs[j]) >= label_part_region_array.shape[1]:
                frac_ymaxs[j] = label_part_region_array.shape[1]
            if int(frac_xmaxs[j]) >= label_part_region_array.shape[2]:
                frac_xmaxs[j] = label_part_region_array.shape[2]

            z_s = int((frac_zmaxs[j] - frac_zmins[j]) / 10)  # 10等分
            y_s = int((frac_ymaxs[j] - frac_ymins[j]) / 10)
            x_s = int((frac_xmaxs[j] - frac_xmins[j]) / 10)
            count_1 = np.zeros(20)
            count_2 = np.zeros(20)
            count_3 = np.zeros(20)

            for z in range(frac_zmins[j], frac_zmaxs[j], z_s):
                for y in range(frac_ymins[j], frac_ymaxs[j], y_s):
                    for x in range(frac_xmins[j], frac_xmaxs[j], x_s):
                        ############ 每次判断六个表面，判断骨折区域和相邻的主体区域，是否同时有像素，若都有，则判断相邻，取相邻最多的类别作为骨折类别 ###########
                        value1 = label_part_region_array[int(frac_zmins[j] - 1), int(y), int(x)]
                        value_frac_1_1 = frac_region_1[j][0, int(y - frac_ymins[j]), int(x - frac_xmins[j])]
                        value_frac_1_2 = frac_region_2[j][0, int(y - frac_ymins[j]), int(x - frac_xmins[j])]
                        value_frac_1_3 = frac_region_3[j][0, int(y - frac_ymins[j]), int(x - frac_xmins[j])]

                        value2 = label_part_region_array[int(frac_zmaxs[j] - 1), int(y), int(x)]
                        value_frac_2_1 = frac_region_1[j][
                            int(frac_zmaxs[j] - frac_zmins[j] - 1), int(y - frac_ymins[j]), int(x - frac_xmins[j])]
                        value_frac_2_2 = frac_region_2[j][
                            int(frac_zmaxs[j] - frac_zmins[j] - 1), int(y - frac_ymins[j]), int(x - frac_xmins[j])]
                        value_frac_2_3 = frac_region_3[j][
                            int(frac_zmaxs[j] - frac_zmins[j] - 1), int(y - frac_ymins[j]), int(x - frac_xmins[j])]

                        value3 = label_part_region_array[int(z), int(frac_ymins[j] - 1), int(x)]
                        value_frac_3_1 = frac_region_1[j][int(z - frac_zmins[j]), 0, int(x - frac_xmins[j])]
                        value_frac_3_2 = frac_region_2[j][int(z - frac_zmins[j]), 0, int(x - frac_xmins[j])]
                        value_frac_3_3 = frac_region_3[j][int(z - frac_zmins[j]), 0, int(x - frac_xmins[j])]

                        value4 = label_part_region_array[int(z), int(frac_ymaxs[j] + 1), int(x)]
                        value_frac_4_1 = frac_region_1[j][
                            int(z - frac_zmins[j]), int(frac_ymaxs[j] - frac_ymins[j] - 1), int(x - frac_xmins[j])]
                        value_frac_4_2 = frac_region_2[j][
                            int(z - frac_zmins[j]), int(frac_ymaxs[j] - frac_ymins[j] - 1), int(x - frac_xmins[j])]
                        value_frac_4_3 = frac_region_3[j][
                            int(z - frac_zmins[j]), int(frac_ymaxs[j] - frac_ymins[j] - 1), int(x - frac_xmins[j])]

                        value5 = label_part_region_array[int(z), int(y), int(frac_xmins[j] - 1)]
                        value_frac_5_1 = frac_region_1[j][int(z - frac_zmins[j]), int(y - frac_ymins[j]), 0]
                        value_frac_5_2 = frac_region_2[j][int(z - frac_zmins[j]), int(y - frac_ymins[j]), 0]
                        value_frac_5_3 = frac_region_3[j][int(z - frac_zmins[j]), int(y - frac_ymins[j]), 0]

                        value6 = label_part_region_array[int(z), int(y), int(frac_xmaxs[j] + 1)]
                        value_frac_6_1 = frac_region_1[j][
                            int(z - frac_zmins[j]), int(y - frac_ymins[j]), int(frac_xmaxs[j] - frac_xmins[j] - 1)]
                        value_frac_6_2 = frac_region_2[j][
                            int(z - frac_zmins[j]), int(y - frac_ymins[j]), int(frac_xmaxs[j] - frac_xmins[j] - 1)]
                        value_frac_6_3 = frac_region_3[j][
                            int(z - frac_zmins[j]), int(y - frac_ymins[j]), int(frac_xmaxs[j] - frac_xmins[j] - 1)]

                        if value1 != 0 and value_frac_1_1 != 0:
                            count_1[value1] += 1
                        if value1 != 0 and value_frac_1_2 != 0:
                            count_2[value1] += 1
                        if value1 != 0 and value_frac_1_3 != 0:
                            count_3[value1] += 1
                        if value2 != 0 and value_frac_2_1 != 0:
                            count_1[value2] += 1
                        if value2 != 0 and value_frac_2_2 != 0:
                            count_2[value2] += 1
                        if value2 != 0 and value_frac_2_3 != 0:
                            count_3[value2] += 1
                        if value3 != 0 and value_frac_3_1 != 0:
                            count_1[value3] += 1
                        if value3 != 0 and value_frac_3_2 != 0:
                            count_2[value3] += 1
                        if value3 != 0 and value_frac_3_3 != 0:
                            count_3[value3] += 1
                        if value4 != 0 and value_frac_4_1 != 0:
                            count_1[value4] += 1
                        if value4 != 0 and value_frac_4_2 != 0:
                            count_2[value4] += 1
                        if value4 != 0 and value_frac_4_3 != 0:
                            count_3[value4] += 1
                        if value5 != 0 and value_frac_5_1 != 0:
                            count_1[value5] += 1
                        if value5 != 0 and value_frac_5_2 != 0:
                            count_2[value5] += 1
                        if value5 != 0 and value_frac_5_3 != 0:
                            count_3[value5] += 1
                        if value6 != 0 and value_frac_6_1 != 0:
                            count_1[value6] += 1
                        if value6 != 0 and value_frac_6_2 != 0:
                            count_2[value6] += 1
                        if value6 != 0 and value_frac_6_3 != 0:
                            count_3[value6] += 1

            region1 = frac_region_1[j].copy()
            region1[region1 == 8] = np.argmax(count_1)
            # print("np.argmax(count_1)", np.argmax(count_1))
            region2 = frac_region_2[j].copy()
            region2[region2 == 9] = np.argmax(count_2)
            # print("np.argmax(count_2)", np.argmax(count_2))
            region3 = frac_region_3[j].copy()
            region3[region3 == 10] = np.argmax(count_3)

            frac_region = region1 + region2 + region3
            fracture_region.append(frac_region)

        for j in range(len(fracture_region)):
            label_part_region_array[frac_zmins[j]:frac_zmaxs[j], frac_ymins[j]:frac_ymaxs[j], frac_xmins[j]:frac_xmaxs[j]] = fracture_region[j]

        ### 为了使得结果依然与最初分成的三个结构对应，则需要进行以下操作，即每个最大的都为类别1，原来类别1现在+1 =1，类别2现在+2=2
        for k in range(10, 0, -1):
            # print(k)
            if k == 1:
                label_part_region_array[label_part_region_array == k] = k + i
            else:
                label_part_region_array[label_part_region_array == k] = k + i + 1

        ### 确保每一次仅在一个部位处理
        ### label_array_part_t[i] 在当前部位内为1，其他为0
        label_part_region_array = label_array_part_t[i] * label_part_region_array

        label_part_add_frac_region_arrays.append(label_part_region_array.astype(int))

        if i == 1:
            right_list = np.unique(label_part_region_array)[1:]
        else:
            left_list = np.unique(label_part_region_array)[1:]

    ################### 将三个分别处理的区域进行合并 ######################
    label_region_basic = label_part_add_frac_region_arrays[0] + label_part_add_frac_region_arrays[1] + label_part_add_frac_region_arrays[2]
    # label_region_basic = label_part_add_frac_region_arrays[1]
    label_rebuild = sitk.GetImageFromArray(label_region_basic)
    label_rebuild.CopyInformation(label)
    label_reguild_path = 'tempt/' + 'rebuild.nii.gz'

    ## 与label统一space和origin
    label_rebuild.SetOrigin(label.GetOrigin())
    label_rebuild.SetSpacing(label.GetSpacing())
    label_rebuild.CopyInformation(label)
    sitk.WriteImage(label_rebuild, label_reguild_path)

    return label_rebuild, label_reguild_path, right_list, left_list