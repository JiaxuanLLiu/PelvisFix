import vtk
import os
import random
import open3d as o3d
import numpy as np
import copy

def show_polydata(poly, ren):
    # 随机颜色显示polydata
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)

    color_R = (random.randint(0, 255)) / 255  # 生成随机颜色
    color_G = (random.randint(0, 255)) / 255
    color_B = (random.randint(0, 255)) / 255

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color_R, color_G, color_B)

    ren.AddActor(actor)

def face_extract(target, source, model):
    # 提取两个碎片中相近的断面， 输出的断面点云按照输入的点云顺序
    fracture1 = display_fracture(source, model, 1.0, False)
    fracture3 = display_fracture(source, target, 5.6)

    fracture2 = display_fracture(target, model, 1.0, False)
    fracture4 = display_fracture(target, source, 5.6)

    face1 = display_fracture(fracture4, fracture2, 1.2)  # 计算断面点云（不属于外表面，且离另一个碎片足够近）
    face2 = display_fracture(fracture3, fracture1, 1.2)

    # face1.paint_uniform_color([0, 1, 0])
    # face2.paint_uniform_color([0, 1, 1])
    # o3d.visualization.draw_geometries([face2, face1], window_name="待配准断面",
    #                                   width=1024, height=768,
    #                                   left=50, top=50,
    #                                   mesh_show_back_face=False)

    return face1, face2

def display_fracture(source, target, dist_thres, inlier=True):
    # 输出断面点云，以健侧为参照 inlier真返回内部点，假返回外部点
    dists = source.compute_point_cloud_distance(target)
    dists = np.asarray(dists)
    ind = np.where(dists < dist_thres)[0]

    inlier_cloud = source.select_by_index(ind)
    outlier_cloud=source.select_by_index(ind, invert=True)

    # outlier_cloud_f, ind_f = outlier_cloud.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.05)
    # outlier_cloud_f, ind_f = outlier_cloud.remove_radius_outlier(nb_points=50,radius=10) # 半径噪声过滤

    # inlier_cloud.paint_uniform_color([0, 1, 0])
    # outlier_cloud.paint_uniform_color([1, 0, 0])
    # # outlier_cloud_filter = open3d_cluster(outlier_cloud)
    #
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
    #                                   window_name="重叠和非重叠点",
    #                                   width=1024, height=768,
    #                                   left=50, top=50,
    #                                   mesh_show_back_face=False)
    if inlier:
        return inlier_cloud
    else:
        return outlier_cloud

def single_ptc(model_poly):    # 读取单个stl文件，并在窗口显示,返回值为提取点云
    """
    Args:
        model_poly:健侧polydata，作为模板
    Returns:
        提取点云
    """
    # reader = vtk.vtkSTLReader()
    # reader.SetFileName(path)
    # reader.Update()

    # polydata = reader.GetOutput()
    #
    # mapper = vtk.vtkPolyDataMapper()
    # mapper.SetInputData(polydata)
    #
    # color_R = (random.randint(0, 255)) / 255  # 生成随机颜色
    # color_G = (random.randint(0, 255)) / 255
    # color_B = (random.randint(0, 255)) / 255
    #
    # actor = vtk.vtkActor()
    # actor.SetMapper(mapper)
    # actor.GetProperty().SetColor(color_R, color_G, color_B)
    #
    # ren.AddActor(actor)  # 在窗口显示

    polydata = model_poly

    pointsvtk = vtk.vtkPoints()
    pointsvtk.DeepCopy(polydata.GetPoints())

    points = np.zeros((pointsvtk.GetNumberOfPoints(), 3))  # 将vtk点云逐点存入数组中
    for n in range(pointsvtk.GetNumberOfPoints()):
        k = np.zeros(3)
        pointsvtk.GetPoint(n, k)
        points[n, 0] = k[0]
        points[n, 1] = k[1]
        points[n, 2] = k[2]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)  # 得到原始模型的点云
    return pcd

def get_mc_contour(file,setvalue):
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
    smoother.SetRelaxationFactor(0.15)  # 越大效果越明显
    smoother.Update()
    return smoother

def stl2ptd_onefile(stl_list):
    # 返回一个点云对象

    ptd_list = []
    pointsvtk = vtk.vtkPoints()
    for i, polydata in enumerate(stl_list):
        points_i = polydata.GetPoints()
        for n in range(points_i.GetNumberOfPoints()):
            point = points_i.GetPoint(n)
            pointsvtk.InsertNextPoint(point)

    points = np.zeros((pointsvtk.GetNumberOfPoints(), 3))  # 将vtk点云逐点存入数组中
    for n in range(pointsvtk.GetNumberOfPoints()):
        k = np.zeros(3)
        pointsvtk.GetPoint(n, k)
        points[n, 0] = k[0]
        points[n, 1] = k[1]
        points[n, 2] = k[2]

    pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pointsvtk)
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd

def stl2ptd_list(stl_list):

    ptd_list = []
    for i, polydata in enumerate(stl_list):
        pointsvtk = vtk.vtkPoints()
        pointsvtk.DeepCopy(polydata.GetPoints())

        points = np.zeros((pointsvtk.GetNumberOfPoints(), 3))  # 将vtk点云逐点存入数组中
        for n in range(pointsvtk.GetNumberOfPoints()):
            k_p = np.zeros(3)
            pointsvtk.GetPoint(n, k_p)
            points[n, 0] = k_p[0]
            points[n, 1] = k_p[1]
            points[n, 2] = k_p[2]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        ptd_list.append(pcd)  # 模型的点云列表

    return ptd_list

def trans_sacrum_ptc(pcd, poly_list):
    # 创建翻转矩阵
    bound = np.zeros(6)
    poly_list[0].GetBounds(bound)
    a = 1
    r = 0.5 * bound[0] + 50
    # ma = np.array([[-1, 0, 0, -10],
    #                [0, 1, 0, 0],
    #                [0, 0, 1, 0],
    #                [0, 0, 0, 1]])

    ma = np.array([[-1, 0, 0, 200],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    pcdT = copy.deepcopy(pcd)
    pcdT.transform(ma)  # 得到翻转后的模板点云

    # 利用icp进行配准
    threshold = 200.0  # 移动范围的阀值
    trans_init = np.asarray([[1, 0, 0, 0],  # 4x4 identity matrix，这是一个转换矩阵，
                             [0, 1, 0, 0],  # 象征着没有任何位移，没有任何旋转，我们输入
                             [0, 0, 1, 0],  # 这个矩阵为初始变换
                             [0, 0, 0, 1]])

    result_icp = o3d.pipelines.registration.registration_icp(
        pcdT, pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    print(result_icp)  # 输出配准的结果准确度等信息
    print("Transformation is:")
    print(result_icp.transformation)  # 打印旋转矩阵

    # pcd.paint_uniform_color([1, 0, 0])
    # pcdT.paint_uniform_color([0, 1, 1])
    # o3d.visualization.draw([pcd, pcdT.transform(result_icp.transformation)])

    return ma, result_icp.transformation


def preprocess_ptc(ptc, voxel_size=2.0):   # 输入点云，返回下采样点云和点云fpfh特征直方图
    pcd_down = o3d.geometry.PointCloud()
    pcd_down = ptc.voxel_down_sample(voxel_size)
    # 计算法向量
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(voxel_size * 2.5, 300))
    # 计算fpfh特征
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(voxel_size * 5, 100))

    return pcd_down, pcd_fpfh

# def rigid_regis(source, target, after_list, distance_threshold = 30*30):
def rigid_regis(source, target, after_list, distance_threshold=10 * 10):
    # 利用fpfh特征直方图进行粗配准，返回值为旋转矩阵
    source_down, source_feature = preprocess_ptc(source)
    target_down, target_feature = preprocess_ptc(target)  # 提取模板点云法向量和特征直方图

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down, target_down, source_feature, target_feature,
                mutual_filter=False,
                max_correspondence_distance=distance_threshold,
                estimation_method=o3d.pipelines.registration.
                TransformationEstimationPointToPoint(False),
                ransac_n=3,
                checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                          o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                   1000000, 0.999))

    # 利用icp进行第二次配准
    # threshold = 200.0  # 移动范围的阀值
    threshold = 50.0  # 移动范围的阀值

    result_icp = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    after_list.append(source_down.transform(result_icp.transformation))

    # print(result_icp)  # 输出配准的结果准确度等信息
    # print("Transformation is:")
    # print(result_icp.transformation)  # 打印旋转矩阵

    # source_down.paint_uniform_color([1, 0, 0])
    # target_down.paint_uniform_color([0, 1, 1])
    # o3d.visualization.draw([source_down.transform(result_icp.transformation), target_down])

    return result_icp.transformation

def icp_regis(source,target, threshold):
    # 利用icp进行第二次配准
    # threshold = 5.0  # 移动范围的阀值
    trans_init = np.asarray([[1, 0, 0, 0],  # 4x4 identity matrix，这是一个转换矩阵，
                             [0, 1, 0, 0],  # 象征着没有任何位移，没有任何旋转，我们输入
                             [0, 0, 1, 0],  # 这个矩阵为初始变换
                             [0, 0, 0, 1]])

    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    # ptc_after_list.append(source_down.transform(result_icp.transformation))

    # print(result_icp)  # 输出配准的结果准确度等信息
    # print("Transformation is:")
    # print(result_icp.transformation)  # 打印旋转矩阵
    #
    # source.paint_uniform_color([1, 0, 0])
    # target.paint_uniform_color([0, 1, 1])
    # o3d.visualization.draw_geometries([source.transform(result_icp.transformation), target])

    return result_icp.transformation

def trans_polydata(polydata:vtk.vtkPolyData, trans):
    # 对polydata进行矩阵变换
    p_matrix = vtk.vtkMatrix4x4()
    for i in range(0, 4):
        for j in range(0, 4):
            p_matrix.SetElement(i, j, trans[i][j])

    p_transform = vtk.vtkTransform()
    p_transform.SetMatrix(p_matrix)

    p_transform_filter = vtk.vtkTransformPolyDataFilter()
    p_transform_filter.SetInputData(polydata)
    p_transform_filter.SetTransform(p_transform)
    p_transform_filter.Update()

    polydata2 = p_transform_filter.GetOutput()
    polydata.DeepCopy(polydata2)

def trans_by_face(k, list, model_sym):
    # 利用断面配准，返回每一个碎片的配准矩阵
    # for k in range(1, len(list)):  # 精配准，基于断面点
        face_self = o3d.geometry.PointCloud()
        face_0 = o3d.geometry.PointCloud()

        for i in range(0, k):
            frac_face1, frac_face2 = face_extract(list[i], list[k], model_sym)
            face_0.points = o3d.utility.Vector3dVector(np.concatenate((np.asarray(face_0.points),
                                                                       np.asarray(frac_face1.points))))  # 拼接点云,其他碎片
            face_self.points = o3d.utility.Vector3dVector(np.concatenate((np.asarray(face_self.points),
                                                                          np.asarray(frac_face2.points))))  # 拼接点云

        # face_self.paint_uniform_color([1, 0, 0])
        # face_0.paint_uniform_color([1, 1, 0])
        # list[k].paint_uniform_color([1, 0, 1])
        # o3d.visualization.draw_geometries([face_self, list[k]], window_name="断面",
        #                                   width=1024, height=768,
        #                                   left=50, top=50,
        #                                   mesh_show_back_face=False)
        icp_ma = icp_regis(face_self, face_0, 5.0)


        return icp_ma

def read_nii(filename):
    '''
    读取nii文件，输入文件路径
    '''
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader

def Nii_Reconstruct_all_label(Image_Path, smoothing_iterations, pass_band, feature_angle):
    #输出一个model list 包含label image中的所有label
    reader1 = read_nii(Image_Path)
    a, b = reader1.GetOutput().GetScalarRange()
    polylist = []

    for j in range(1, int(b)+1):
        contour = get_mc_contour(reader1, j)
        smoothe = smoothing(smoothing_iterations, pass_band, feature_angle, contour)
        polydata = vtk.vtkPolyData()
        polydata = smoothe.GetOutput()
        polylist.append(polydata)  # 将所有model存入list中

    return polylist

def pelvis_regis(rebuild_path, right_list_cla, left_list_cla):
    reduction_poly_list = []
    pelvis_polydata_list = Nii_Reconstruct_all_label(rebuild_path, 50, 0.005, 180)
    right_list = []
    left_list = []
    sacrum_list = []
    polylist = []

    ## 提取左右两侧以及骶骨的stl list
    for i in range(0, len(right_list_cla)):
        cla = right_list_cla[i] - 1
        right_list.append(pelvis_polydata_list[cla])

    for i in range(0, len(left_list_cla)):
        cla = left_list_cla[i] - 1
        left_list.append(pelvis_polydata_list[cla])

    sacrum_list.append(pelvis_polydata_list[0])

    if len(left_list) <= len(right_list):
        ptclist = stl2ptd_list(right_list)  # ptclist为患侧碎片点云列表
        polylist = right_list  # polylist为患侧碎片模型列表
        model_ptc = stl2ptd_onefile(left_list)  # model_ptc为健侧点云对象

    else:
        ptclist = stl2ptd_list(left_list)
        polylist = left_list
        model_ptc = stl2ptd_onefile(right_list)

    sacrum_ptc = stl2ptd_onefile(sacrum_list)  # 提取骶骨点云
    ma1, trans_ma = trans_sacrum_ptc(sacrum_ptc, sacrum_list)  # 提取翻转和镜像配准矩阵

    ptc_after_list = []  # 复位后的point cloud list(降采样后)

    model_sym = copy.deepcopy(model_ptc)
    model_sym.transform(ma1)
    model_sym.transform(trans_ma)  # 健侧镜像点云

    tra_matrix = []  # 配准矩阵list
    # 粗配准
    for k in range(0, len(polylist)):  # 对每一个碎片进行配准
        trans = rigid_regis(ptclist[k], model_sym, ptc_after_list)  # 粗配准
        trans_polydata(polylist[k], trans)

        tra = np.asarray(trans)
        tra_matrix.append(tra)

    # 断面配准
    for k in range(1, len(polylist)):
        icp_ma = trans_by_face(k, ptc_after_list, model_sym)
        ptc_after_list[k] = ptc_after_list[k].transform(icp_ma)
        trans_polydata(polylist[k], icp_ma)

        tra2 = np.asarray(icp_ma)
        tra_matrix[k] = np.matmul(tra_matrix[k], tra2)

    # 合成一个polydata
    appendfilter = vtk.vtkAppendPolyData()
    for k in range(0, len(polylist)):
        # show_polydata(fracture_poly_list[k], ren)
        appendfilter.AddInputData(polylist[k])
        appendfilter.Update()

    # 移除冗余点
    cleanfilter = vtk.vtkCleanPolyData()
    cleanfilter.SetInputConnection(appendfilter.GetOutputPort())
    cleanfilter.Update()

    reduction_poly_list.append(cleanfilter.GetOutput())

    # 导出断面点云
    frac_pcds = []
    for k in range(0, len(ptc_after_list)):
        face_self = o3d.geometry.PointCloud()
        face_0 = o3d.geometry.PointCloud()

        for i in range(0, len(ptc_after_list)):
            if i != k:
                frac_face1, frac_face2 = face_extract(ptc_after_list[i], ptc_after_list[k], model_sym)
                face_0.points = o3d.utility.Vector3dVector(np.concatenate((np.asarray(face_0.points),
                                                                           np.asarray(
                                                                               frac_face1.points))))  # 拼接点云,其他碎片
                face_self.points = o3d.utility.Vector3dVector(np.concatenate((np.asarray(face_self.points),
                                                                              np.asarray(
                                                                                  frac_face2.points))))  # 拼接点云
                frac_pcds.append(face_self)

        # path_new = "tempt/" + str(k) + ".pcd"
        # o3d.io.write_point_cloud(path_new, face_self)

    # # 导出配准后的stl文件
    # for k in range(0, len(polylist)):
    #     stlWriter = vtk.vtkSTLWriter()
    #     stlWriter.SetFileName("fra" + str(k) + ".stl")
    #     stlWriter.SetInputData(polylist[k])
    #     stlWriter.Write()

    # 两侧复位结果合成一个polydata
    appendfilter = vtk.vtkAppendPolyData()
    for k in range(len(reduction_poly_list)):
        # show_polydata(fracture_poly_list[k], ren)
        appendfilter.AddInputData(reduction_poly_list[k])
        appendfilter.Update()

    # 移除冗余点
    cleanfilter = vtk.vtkCleanPolyData()
    cleanfilter.SetInputConnection(appendfilter.GetOutputPort())
    cleanfilter.Update()

    print("len(frac_pcds)", len(frac_pcds))
    print("len(polylist)", len(polylist))
    return cleanfilter.GetOutput(), polylist, frac_pcds
