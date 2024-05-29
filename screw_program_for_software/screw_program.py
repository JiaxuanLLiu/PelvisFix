# coding = utf-8
import numpy as np
import screw_program_for_software.data_preprocess_software as data_preprocess_software
import vtk
import open3d as o3d

def vtk_to_pcd(polydata_list):
    pcd_list = []
    pointsvtk = vtk.vtkPoints()
    for polydata in polydata_list:
        pointsvtk.DeepCopy(polydata.GetPoints())

        points = np.zeros((pointsvtk.GetNumberOfPoints(), 3)) 
        for n in range(pointsvtk.GetNumberOfPoints()):
            k_p = np.zeros(3)
            pointsvtk.GetPoint(n, k_p)
            points[n, 0] = k_p[0]
            points[n, 1] = k_p[1]
            points[n, 2] = k_p[2]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd_list.append(pcd)  

    return pcd_list

def screw_program(stls, frac_pcds):
    if len(frac_pcds) == 0:
        return
  
    all_pcds = vtk_to_pcd(stls)

    rest_pcds = data_preprocess_software.get_rest_pcds(all_pcds, frac_pcds)
    rest_pcds = data_preprocess_software.downSample(rest_pcds)

    screw_actors = []

    # join one polydata
    appendfilter = vtk.vtkAppendPolyData()
    for k in range(0, len(screw_actors)):
        # show_polydata(fracture_poly_list[k], ren)
        appendfilter.AddInputConnection(screw_actors[k].GetOutputPort())
        appendfilter.Update()

    # remove additional points
    cleanfilter = vtk.vtkCleanPolyData()
    cleanfilter.SetInputConnection(appendfilter.GetOutputPort())
    cleanfilter.Update()

    # return screw_info, screw_actors
    return cleanfilter.GetOutput()