import os
import vtk
import torch
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm
import SimpleITK as sitk
import matplotlib.pyplot as plt
from networks.Siamese_regression import Siamese
from scipy.spatial.transform import Rotation as R

"""
    fracture zone identification 
"""
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda', 1)

random_seed = 123
torch.manual_seed(random_seed)

def resize_image(itkimage_array, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    """
   RESAMPLE
    :param itkimage_array:
    :param newSize:such as [1,1,1]
    :param resamplemethod:
    :return:
    """
    itkimage = sitk.GetImageFromArray(itkimage_array)
    newSize = np.array(newSize, float)
    originSpcaing = itkimage.GetSpacing()
    Resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    factor = newSize / originSize
    newSize = newSize.astype(int)
    newSpacing = originSpcaing / factor

    Resampler.SetReferenceImage(itkimage)
    Resampler.SetOutputSpacing(newSpacing.tolist())
    Resampler.SetSize(newSize.tolist())

    Resampler.SetInterpolator(resamplemethod)
    imgResampled = Resampler.Execute(itkimage)

    imgResampled_array = sitk.GetArrayFromImage(imgResampled)
    return imgResampled_array, imgResampled

def Siamese_filter(Image_candidate_area_array, Image_opposide_candidate_area_array, model, input_shape):
    """
    输入通过聚类得到的可能的断裂区域
    通过孪生网络进行筛选
    :param Image_candidate_area_array:
    :param Image_opposide_candidate_area_array:
    :param model:trained model
    :param input_shape：
    :return: Similar 1，Different 0
    """
    Image_1_array, _ = resize_image(Image_candidate_area_array, input_shape, resamplemethod=sitk.sitkNearestNeighbor)
    Image_1_array = np.expand_dims(np.array(Image_1_array), 0)  # 补一维通道数
    Image_1_array = np.expand_dims(np.array(Image_1_array), 0)  # 补一维batch

    Image_2_array, _ = resize_image(Image_opposide_candidate_area_array, input_shape, resamplemethod=sitk.sitkNearestNeighbor)
    Image_2_array = np.expand_dims(np.array(Image_2_array), 0)  # 补一维通道数
    Image_2_array = np.expand_dims(np.array(Image_2_array), 0)  # 补一维batch

    with torch.no_grad():
        input_1 = torch.from_numpy(Image_1_array).type(torch.FloatTensor)
        input_2 = torch.from_numpy(Image_2_array).type(torch.FloatTensor)

        input_1 = input_1.to(device)
        input_2 = input_2.to(device)

        outputs_x1_p, outputs_x2_p, outputs_abs, outputs_x1_t, outputs_x2_t  = model([input_1, input_2])

        ######################## 综合判断 ##########################
        output_BCE = torch.nn.Sigmoid()(outputs_abs)

        outputs_x1_t = torch.nn.Sigmoid()(outputs_x1_t)
        predict_x1_t = outputs_x1_t.cpu().detach().numpy()[0][0]
        predict_x1 = round(predict_x1_t)

        outputs_x2_t = torch.nn.Sigmoid()(outputs_x2_t)
        predict_x2_t = outputs_x2_t.cpu().detach().numpy()[0][0]
        predict_x2 = round(predict_x2_t)

        if predict_x1 > 1:
            predict_x1 = 1
        elif predict_x1 < 0:
            predict_x1 = 0
        if predict_x2 > 1:
            predict_x2 = 1
        elif predict_x2 < 0:
            predict_x2 = 0

        if (round(output_BCE.cpu().detach().numpy()[0][0]) == 1 and round(predict_x1) == 0 and round(
                predict_x2) == 0) or (
                round(output_BCE.cpu().detach().numpy()[0][0]) == 1 and round(predict_x1) == 0 and round(
            predict_x2) == 1) or (
                round(output_BCE.cpu().detach().numpy()[0][0]) == 0 and round(predict_x1) == 0 and round(
            predict_x2) == 0) or (
                round(output_BCE.cpu().detach().numpy()[0][0]) == 0 and round(predict_x1) == 0 and round(
            predict_x2) == 1):
            isFracture = 0
        else:
            isFracture = 1

    return isFracture

#重采样设置CT的大小（size），切片比例（new_spacing）随着改变
def ImageResample_size(sitk_image, new_size = [256, 300, 256], is_label = False):
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

def read_nii(filename):
    '''
    读取nii文件，输入文件路径
    '''
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader


def get_mc_contour(file, setvalue):
    '''
    计算轮廓的方法
    file:读取的vtk类
    setvalue:要得到的轮廓的值
    '''
    contour = vtk.vtkDiscreteMarchingCubes()
    contour.SetInputConnection(file.GetOutputPort())

    contour.ComputeNormalsOn()
    contour.SetValue(0, setvalue)
    return contour


def smoothing(smoothing_iterations, pass_band, feature_angle, contour):
    '''
    使轮廓变平滑
    smoothing_iterations:迭代次数
    pass_band:值越小单次平滑效果越明显
    feature_angle:暂时不知道作用
    '''
    # vtk有两种平滑函数，效果类似
    vtk.vtkSmoothPolyDataFilter()
    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputConnection(contour.GetOutputPort())
    smoother.SetNumberOfIterations(20)
    smoother.SetRelaxationFactor(0.15)    # 越大效果越明显
    smoother.Update()
    return smoother


def Nii_Reconstruct_STL(Image_Path, smoothing_iterations, pass_band, feature_angle, save_stlPath):
    reader1 = read_nii(Image_Path)
    contour = get_mc_contour(reader1, 1)
    smoothe = smoothing(smoothing_iterations, pass_band, feature_angle, contour)
    stlWriter = vtk.vtkSTLWriter()
    stlWriter.SetFileName(save_stlPath)
    stlWriter.SetInputConnection(smoothe.GetOutputPort())
    # stlWriter.SetInputConnection(contour.GetOutputPort()) #不做平滑
    stlWriter.Write()
    # STL = contour.GetOutput() # 不做平滑
    STL = smoothe.GetOutput()
    return STL

def read_STL(filepath):
    # load stl
    mesh_stl = o3d.io.read_triangle_mesh(filepath)
    mesh_stl.compute_vertex_normals()
    # V_mesh 为stl网格的顶点坐标序列，shape=(n,3)，这里n为此网格的顶点总数，其实就是浮点型的x,y,z三个浮点值组成的三维坐标
    V_mesh = np.asarray(mesh_stl.vertices)
    # stl/ply -> pcd
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(V_mesh)
    return pcd

def get_STL(pcd, write_path):
    V_mesh = np.asarray(pcd.vertices)
    F_mesh = np.asarray(pcd.triangles)

    mesh_stl = o3d.geometry.TriangleMesh()
    mesh_stl.vertices = o3d.utility.Vector3dVector(V_mesh)
    mesh_stl.triangles = o3d.utility.Vector3iVector(F_mesh)
    mesh_stl.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_stl], window_name="stl")
    o3d.io.write_triangle_mesh(write_path, mesh_stl)


def ICP_STL_Registration(transormer_stl, target_stl, Iterations):
    """
    :param transormer_stl: 变换的stl
    :param target_stl: 目标stl
    :param Iterations: 迭代次数
    :return: 变换矩阵
    """
    # 原数据
    transormer = vtk.vtkPolyData()
    transormer.SetPoints(transormer_stl.GetPoints())

    # 目标数据
    target = vtk.vtkPolyData()
    target.SetPoints(target_stl.GetPoints())

    transormerGlypyFilter = vtk.vtkVertexGlyphFilter()
    transormerGlypyFilter.SetInputData(transormer)
    transormerGlypyFilter.Update()

    targetGlyphFilter = vtk.vtkVertexGlyphFilter()
    targetGlyphFilter.SetInputData(target)
    targetGlyphFilter.Update()

    # 进行ICP配准求变换矩阵
    icpTransform = vtk.vtkIterativeClosestPointTransform()
    icpTransform.SetSource(transormerGlypyFilter.GetOutput())  # 设置原点集
    icpTransform.SetTarget(targetGlyphFilter.GetOutput())  # 目标点集
    icpTransform.GetLandmarkTransform().SetModeToRigidBody()  # 计算ICP迭代中的最佳匹配点集
    icpTransform.SetMaximumNumberOfIterations(Iterations)  # 用于设置ICP算法迭代的次数
    icpTransform.StartByMatchingCentroidsOn()  # 设置配准之前先计算两个点集中心，并平移源点集使得两者重心重合
    icpTransform.Modified()
    Matrix = icpTransform.GetMatrix()
    # print(icpTransform.GetMatrix())  # 获取变换矩阵
    icpTransform.Update()

    return Matrix

def DBSCAN_cluster(downpcd, eps=3.0, min_points=100):
    """
    :param downpcd: 降采样后的点云
    :param eps: 同一聚类中最大点间距
    :param min_points: 有效聚类的最小点数
    :return:
    """
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(downpcd.cluster_dbscan(eps, min_points, print_progress=True))
    max_label = labels.max()  # 获取聚类标签的最大值 [-1,0,1,2,...,max_label]，label = -1 为噪声，因此总聚类个数为 max_label + 1
    # print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # labels = -1 的簇为噪声，以黑色显示
    downpcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return downpcd, labels

def divide_poing_cloud(downpcd, labels):
    """
    :param downpcd: 降采样点云
    :param labels: 聚类后的各个label
    :return: 存储合并后点云的list，存储显示目标的list
    """
    min = labels.min()
    max = labels.max()
    ##### 提取背景 ####
    background_index = np.where(labels == min)
    background_pcd = downpcd.select_by_index(np.array(background_index)[0])  # 根据下标提取点云点
    background_pcd.paint_uniform_color([0.09, 0.15, 0.15])
    Target_pcd_list = []
    Visualiz_list = []
    Visualiz_list.append(background_pcd)

    ##### 提取出非背景点云 ####
    for label in range(min + 1, max + 1):
        label_index = np.where(labels == label)  # 提取分类为label的聚类点云下标
        label_pcd = downpcd.select_by_index(np.array(label_index)[0])  # 根据下标提取点云点
        # print('label: ', str(label), '点云数：', len(label_pcd.points))
        if label == min + 1:
            Target_pcd_list.append(label_pcd)
            Visualiz_list.append(label_pcd)
        else:
            ###### 对质心小于阈值的点云合并 #####
            is_merge = False
            for i in range(len(Target_pcd_list)):
                center = label_pcd.get_center()  # 当前点云质心
                center_pr = Target_pcd_list[i].get_center()  # 当前点云质心
                distance = (center[0] - center_pr[0]) ** 2 + (center[1] - center_pr[1]) ** 2 + (
                        center[2] - center_pr[2]) ** 2
                if distance < 800:
                    Target_pcd_list[i] += label_pcd
                    is_merge = True
                    continue

            if is_merge == False:
                Target_pcd_list.append(label_pcd)

    return background_pcd, Target_pcd_list, Visualiz_list

def is_region_overlap(box1, box2):
    """
    判断两个矩形是否相交
    box=(xA,yA,zA,xB,yB,zB)
    """

    x01, y01, z01, x02, y02, z02 = box1
    x11, y11, z11, x12, y12, z12 = box2

    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
    lz = abs((z01 + z02) / 2 - (z11 + z12) / 2)
    sax = abs(x01 - x02)
    sbx = abs(x11 - x12)
    say = abs(y01 - y02)
    sby = abs(y11 - y12)
    saz = abs(z01 - z02)
    sbz = abs(z11 - z12)
    if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2 and lz <= (saz + sbz):
        return True
    else:
        return False


def cal_IOU(box1, box2):
    """
     box=(xA,yA,zA,xB,yB,zB)
     计算两个矩形框的重合度
    """

    if is_region_overlap(box1, box2) == True:
        x01, y01, z01, x02, y02, z02 = box1
        x11, y11, z11, x12, y12, z12 = box2
        x_c = min(x02, x12) - max(x01, x11)
        y_c = min(y02, y12) - max(y01, y11)
        z_c = min(z02, z12) - max(z01, z11)
        intersection = x_c * y_c * z_c
        area1 = (x02 - x01) * (y02 - y01) * (z02 - z01)
        area2 = (x12 - x11) * (y12 - y11) * (z12 - z11)
        coincide = intersection / (area1 + area2 - intersection)
        return coincide
    else:
        return 0

def frac_zone_detect(Image, Label):
    """
    读取Image, label
    进行骨折裂痕区域检测
    保存骨折区域的image和label
    return 骨折区域图像list,aabb的顶点坐标array，obb的顶点坐标array
    """

    input_shape = [64, 64, 64]

    Label_array = sitk.GetArrayFromImage(Label)
    ################################# 网络参数设定 ###################################
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "model/dist bce reg VGG-self-attention batch4 size64/30.pth"
    ################################# 加载模型 ###################################
    model = Siamese(input_shape)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    num_cases = 0

    boundbox_index_array = [] # obb包围盒顶点坐标，用来进行配准
    position_array = [] # aabb包围盒顶点坐标，用来进行复原
    frac_list = []  # 保存检测出的骨折块Label

    num_frac_list = [0, 0]

    for dir in range(2):
        ################ dir = 0 -> 检测左侧骨折 ###############
        ################ dir = 1 -> 检测右侧骨折 ###############
        Image_array = sitk.GetArrayFromImage(Image)
        Image.SetOrigin([0, 0, 0])

        Image.SetOrigin(Label.GetOrigin())
        Image.SetSpacing(Label.GetSpacing())

        ########################### 获取骨盆左右骨头单独的label ########################
        Label_left_array = sitk.GetArrayFromImage(Label)
        Label_right_array = sitk.GetArrayFromImage(Label)

        if dir == 0: ### 检测左侧骨折
            Label_left_array[Label_left_array != 3] = 0
            Label_left_array[Label_left_array == 3] = 1
            Label_right_array[Label_right_array != 2] = 0
            Label_right_array[Label_right_array == 2] = 1
        elif dir == 1: ### 检测右侧骨折
            Label_left_array[Label_left_array != 2] = 0
            Label_left_array[Label_left_array == 2] = 1
            Label_right_array[Label_right_array != 3] = 0
            Label_right_array[Label_right_array == 3] = 1

        Label_left = sitk.GetImageFromArray(Label_left_array)
        Label_left.SetOrigin(Label.GetOrigin())
        Label_left.SetSpacing(Label.GetSpacing())
        Label_right = sitk.GetImageFromArray(Label_right_array)
        Label_right.SetOrigin(Label.GetOrigin())
        Label_right.SetSpacing(Label.GetSpacing())

        sitk.WriteImage(Label_left, "tempt/Label_left.nii.gz")
        # sitk.WriteImage(Label_right, "tempt/Label_right.nii.gz")

        ########################### 将右边label对称到左侧 ########################
        Label_right_Mirror = sitk.Flip(Label_right, sitk.VectorBool([True, False, False]), True)
        Label_right_Mirror.SetOrigin([0, 0, 0])
        Label_right_Mirror.SetSpacing(Label.GetSpacing())
        Label_right_Mirror.SetDirection(Label.GetDirection())
        sitk.WriteImage(Label_right_Mirror, 'tempt/Label_right_Mirror.nii.gz')

        ########################### 对原图做镜像 #################################
        Image_Mirror = sitk.Flip(Image, sitk.VectorBool([True, False, False]), True)
        Image_Mirror.SetOrigin([0, 0, 0])
        Image_Mirror.SetSpacing(Label.GetSpacing())
        Image_Mirror.SetDirection(Label.GetDirection())
        # sitk.WriteImage(Image_Mirror, 'tempt/image_Mirror.nii.gz')

        ########################### 对左右骨盆三维重建 ############################
        stl_Left = Nii_Reconstruct_STL("tempt/Label_left.nii.gz", 50, 0.005, 180, 'tempt/Step1_Left_STL.stl')
        stl_Right_mirror = Nii_Reconstruct_STL("tempt/Label_right_Mirror.nii.gz", 50, 0.005, 180, 'tempt/Step1_Right_Mirror_STL.stl')

        ################################# ICP变换 ##############################
        Matrix = ICP_STL_Registration(stl_Right_mirror, stl_Left, 50)

        ##################### 利用变换矩阵对原图及label进行变换 ############################
        rot_matrix = [[Matrix.GetElement(0, 0), Matrix.GetElement(0, 1), Matrix.GetElement(0, 2)],
                      [Matrix.GetElement(1, 0), Matrix.GetElement(1, 1), Matrix.GetElement(1, 2)],
                      [Matrix.GetElement(2, 0), Matrix.GetElement(2, 1), Matrix.GetElement(2, 2)]]

        # 进行旋转变换
        rot = R.from_matrix(rot_matrix)
        trans = sitk.Similarity3DTransform()
        trans.SetRotation(rot.as_quat())

        # 进行平移变换
        trans.SetTranslation([Matrix.GetElement(0, 3), Matrix.GetElement(1, 3), Matrix.GetElement(2, 3)])

        Image_Mirror_Transform = sitk.Resample(Image_Mirror, Image_Mirror, trans.GetInverse(), sitk.sitkNearestNeighbor)
        # sitk.WriteImage(Image_Mirror_Transform, "tempt/Image_Mirror_Transform.nii.gz")

        ########################### 读取stl进行点云操作 ##########################
        pcd_Left = read_STL("tempt/Step1_Left_STL.stl")

        ############################## 降采样 ##################################
        print("->正在体素下采样...")
        voxel_size = 0.5
        downpcd = pcd_Left.voxel_down_sample(voxel_size)

        ############################## DBSCAN聚类 ##############################
        print("->正在DBSCAN聚类...")
        # eps = 2.9      # 同一聚类中最大点间距
        # eps = 3.2  # 同一聚类中最大点间距
        # eps = 3.1  # 同一聚类中最大点间距
        eps = 3.0  # 同一聚类中最大点间距
        min_points = 100     # 有效聚类的最小点数
        downpcd_cluster, labels = DBSCAN_cluster(downpcd, eps, min_points)
        ############################## 根据索引提取点云 ###########################
        print("->正在根据索引提取点云...")
        background_pcd, Target_pcd_list, Visualiz_list = divide_poing_cloud(downpcd_cluster, labels)

        frac_region_list = []  # 用于存储每一例中骨折的检测块坐标
         ############################### 计算包围盒 ###############################
        for i in range(len(Target_pcd_list)):
            # Visualiz_list.append(Target_pcd_list[i])
            aabb = Target_pcd_list[i].get_axis_aligned_bounding_box()
            # print("aabb points", np.array(aabb.get_box_points()))
            obb = Target_pcd_list[i].get_oriented_bounding_box()
            obb.color = (0, 1, 0)
            index_obb = obb.get_box_points()
            center_obb = obb.get_center()

            center = aabb.get_center()
            # print("center", center)
            extent = aabb.get_extent()
            aabb.color = (1, 0, 0)

            ########################## 计算所需提取的区域坐标 ####################
            origin = Image.GetOrigin()
            space = Image.GetSpacing()
            xmin = int((center[0] - extent[0] / 2) / space[0])
            xmax = int((center[0] + extent[0] / 2) / space[0])
            ymin = int((center[1] - extent[1] / 2) / space[1])
            ymax = int((center[1] + extent[1] / 2) / space[1])
            zmin = int((center[2] - extent[2] / 2) / space[2])
            zmax = int((center[2] + extent[2] / 2) / space[2])

            x = xmax - xmin
            y = ymax - ymin
            z = zmax - zmin

            ## 过滤太小的区域
            if x * y * z < 8 * 8 * 8:
                Result = 0

            ## 太小的区域放大三倍
            if x * y * z < 15 * 15 * 15:
                xmax = int(xmax + x)
                xmin = int(xmin - x)
                ymax = int(ymax + y)
                ymin = int(ymin - y)
                zmax = int(zmax + z)
                zmin = int(zmin - z)

            ## 小的区域放大一倍
            if x * y * z < 25 * 25 * 20 and x * y * z >= 15 * 15 * 15:
                xmax = int(xmax + x / 2)
                xmin = int(xmin - x / 2)
                ymax = int(ymax + y / 2)
                ymin = int(ymin - y / 2)
                zmax = int(zmax + z / 2)
                zmin = int(zmin - z / 2)

            ######### 保存检测区域 ##########
            Image_candidate_area_array = Image_array[zmin:zmax, ymin:ymax, xmin:xmax]
            Image_candidate_area = sitk.GetImageFromArray(Image_candidate_area_array)

            ######### 保存另一侧区域 ##########
            Image_Mirror_Transform_array = sitk.GetArrayFromImage(Image_Mirror_Transform)
            Image_opposide_candidate_area_array = Image_Mirror_Transform_array[zmin:zmax, ymin:ymax, xmin:xmax]

            Result = Siamese_filter(Image_candidate_area_array, Image_opposide_candidate_area_array, model, input_shape)

            if Result == 1:
                ## 检测到属于骨折的区域，赋上其他颜色
                # print("Frecture:", str(num_cases))
                colors = plt.get_cmap("tab20")(i)
                Target_pcd_list[i].paint_uniform_color(colors[:3])
                Visualiz_list.append(Target_pcd_list[i])
                # Visualiz_list.append(aabb)
                Visualiz_list.append(obb)

                ##### 统计骨折数量 #####
                num_frac_list[dir] += 1

                ######### 计算该区域与先前其他区域是否有重合，若不重合则保存该区域 ##########
                coordinate = [xmin, ymin, zmin, xmax, ymax, zmax]
                overlap = 0
                for k in range(len(frac_region_list)):
                    coincide = cal_IOU(coordinate, frac_region_list[k])
                    # print("coincidene:", coincide)
                    # if coincide > 0.3:
                    if coincide > 0.8:
                        overlap = 1
                if overlap == 1:
                    continue
                else:
                    frac_region_list.append(coordinate)

                ######### 保存骨折块的Label ##########
                Label_candidate_area_array = Label_array[zmin:zmax, ymin:ymax, xmin:xmax]
                Label_candidate_area = sitk.GetImageFromArray(Label_candidate_area_array)
                frac_list.append(Label_candidate_area)

                position_array.append([xmin, xmax, ymin, ymax, zmin, zmax])

                index_obb = np.array(index_obb).astype(int)
                center_obb = np.array(center_obb).astype(int)
                space = Image.GetSpacing()

                boundbox_index_array.append([index_obb[0][0], index_obb[0][1], index_obb[0][2],
                                                index_obb[1][0], index_obb[1][1], index_obb[1][2],
                                                index_obb[2][0], index_obb[2][1], index_obb[2][2],
                                                index_obb[3][0], index_obb[3][1], index_obb[3][2],
                                                index_obb[4][0], index_obb[4][1], index_obb[4][2],
                                                index_obb[5][0], index_obb[5][1], index_obb[5][2],
                                                index_obb[6][0], index_obb[6][1], index_obb[6][2],
                                                index_obb[7][0], index_obb[7][1], index_obb[7][2]])

            else:
                ## 检测到不属于骨折的区域，赋上背景颜色
                # print("No fracture:", str(num_cases))
                # Target_pcd_list[i].paint_uniform_color([0, 0, 0])
                Target_pcd_list[i].paint_uniform_color([0.09, 0.15, 0.15])
                Visualiz_list.append(Target_pcd_list[i])

            num_cases = num_cases + 1

        # # 可视化
        # o3d.visualization.draw_geometries(Visualiz_list, "Predict")

    boundbox_index_array = np.array(boundbox_index_array)
    position_array = np.array(position_array)
    return frac_list, position_array, boundbox_index_array, num_frac_list



