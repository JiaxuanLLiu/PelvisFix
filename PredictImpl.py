import os
import vtk
import torch
import SimpleITK as sitk
from Separation_methods.Select_all import frac_zone_detect
from Separation_methods.seg_frac_watershed import seg_frac_by_watershed
from rebuild_from_segmentation import rebuid_from_seg
# from registration import Reduction_by_frac_register
from Separation_methods.Pelvis_segmentation import pelvis_segmentation
from Reduction_methods import pelvis_regis
from screw_program_for_software.screw_program import screw_program

device = torch.device('cuda', 1)

def main(input_path, output_seg_path, output_model_path, output_model_screw_path):

    reader = sitk.ImageFileReader()
    reader.SetFileName(input_path)
    input_image = reader.Execute()

    predict_label = pelvis_segmentation(input_image)
    predict_label.CopyInformation(input_image)

    frac_list, position_array, boundbox_index_array, num_frac_list = frac_zone_detect(input_image, predict_label)
    frac_seg_list = seg_frac_by_watershed(frac_list)
    segmentation, label_reguild_path, right_list, left_list = rebuid_from_seg(frac_seg_list, predict_label, position_array)
    output_poly, polylist, frac_pcds = pelvis_regis(label_reguild_path, right_list, left_list)
    output_screw_poly = screw_program(polylist, frac_pcds)

    segmentation.CopyInformation(input_image)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(output_seg_path)
    writer.Execute(segmentation)

    """
    临时修正，应用平移
    """
    transform = vtk.vtkTransform()
    transform.Translate(10, 0, 0)  # 沿X轴平移10个单位

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputData(output_poly)
    transformFilter.Update()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_model_path)
    writer.SetDataModeToBinary()
    writer.SetInputData(transformFilter.GetOutput())
    writer.Write()

    transformFilter_1 = vtk.vtkTransformPolyDataFilter()
    transformFilter_1.SetTransform(transform)
    transformFilter_1.SetInputData(output_screw_poly)
    transformFilter_1.Update()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_model_screw_path)
    writer.SetDataModeToBinary()
    writer.SetInputData(transformFilter_1.GetOutput())
    writer.Write()





