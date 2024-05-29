# coding = utf-8
import screw_program_for_software.screw_setting as screw_setting
import numpy as np
import open3d as o3d
from tqdm import tqdm
import scipy.spatial as spatial


def remove_outliers(pcds, nd=screw_setting.nd, std_rt=screw_setting.std_rt):
    new_pcds = []
    for pcd in pcds:
        n_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nd, std_ratio=std_rt)
        new_pcds.append(n_pcd)
    return new_pcds


def downSample(pcds, voxel_size=screw_setting.voxel_size):
    dsp_pcds = []
    for i in range(len(pcds)):
        pcd = pcds[i]
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        dsp_pcds.append(pcd)
    return dsp_pcds


def pcds_normals_outside(pcds):
    all_points = np.empty((0, 3))
    all_normals = np.empty((0, 3))
    allPoints = []
    allNormals = []
    sizes = [0]
    for i in range(len(pcds)):
        pcd = pcds[i]
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))
        all_points = np.concatenate([all_points, np.array(pcd.points)], axis=0)
        all_normals = np.concatenate([all_normals, np.array(pcd.normals)], axis=0)
        allPoints.append(np.array(pcd.points))
        allNormals.append(np.array(pcd.normals))
        sizes.append(sizes[i] + np.array(pcd.points).shape[0])
    z_axis = np.array([0, 0, 1])
    dsbs = np.dot(all_points, z_axis.T)
    index = np.argmax(dsbs).flatten()[0]
    tmp_n = None
    for i in range(len(sizes)):
        if index == 0:
            if np.dot(all_normals[index], z_axis.T) <= 0:
                allNormals[0][0] = -allNormals[0][0]
            tmp_n = allNormals[0][0].copy()
            break
        elif index < sizes[i]:
            if np.dot(all_normals[index], z_axis.T) <= 0:
                allNormals[i-1][index-sizes[i-1]] = -allNormals[i-1][index-sizes[i-1]] 
            tmp_n = allNormals[i-1][index-sizes[i-1]].copy()
            break
    tmp_points = all_points.copy()
    tmp_p = tmp_points[index].copy()
    tmp_points[index] = np.array([10000, 10000, 10000])
    for i in tqdm(range(all_points.shape[0]), desc="\033[31mNormalizing direction:\033[0m",):
        tmp_norm = np.linalg.norm(tmp_points - np.expand_dims(tmp_p, 0).repeat(tmp_points.shape[0], axis=0), axis=1)
        tmp_idx = np.argmin(tmp_norm).flatten()[0]
        tmp_p = tmp_points[tmp_idx].copy()
        tmp_points[tmp_idx] = np.array([10000, 10000, 10000])
        for i in range(len(sizes)):
            if tmp_idx == 0:
                if np.dot(allNormals[0][0], tmp_n.T) <= 0:
                    allNormals[0][0] = -allNormals[0][0]
                tmp_n = allNormals[0][0].copy()
                break
            elif index < sizes[i]:
                if np.dot(allNormals[i-1][index-sizes[i-1]], tmp_n.T) <= 0:
                    allNormals[i-1][index-sizes[i-1]] = -allNormals[i-1][index-sizes[i-1]] 
                tmp_n = allNormals[i-1][index-sizes[i-1]].copy()
                break
    new_pcds = []
    for i in range(len(allPoints)):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(allPoints[i])
        pcd.normals = o3d.utility.Vector3dVector(allNormals[i])
        new_pcds.append(pcd)
    # visualization.points_visualization_by_vtk(new_pcds)
    return new_pcds

        
def get_rest_pcds(all_pcds, frac_pcds, radius=screw_setting.screw_radius - 0.5):
    all_points = []
    frac_points = []
    trees = []
    rest_points = []
    for frac_pcd in frac_pcds:
        frac_points.append(np.asarray(frac_pcd.points))
    for all_pcd in all_pcds:
        all_points.append(np.array(all_pcd.points))
        trees.append(spatial.KDTree(np.array(all_pcd.points)))
        rest_points.append(0)
    for i in range(len(frac_points)): #tqdm(range(len(frac_points)), desc="\033[31mGenerating effect point clouds:\033[0m",):
        frac_ps = frac_points[i]
        index = None
        min_dist = 1000
        indices = np.empty((0, 1))
        for j in range(len(all_points)):
            all_ps = all_points[j]
            _, tmp_indices = trees[j].query(frac_ps, 1, workers=-1)
            tmp_indices = np.array(tmp_indices).flatten()
            dist = np.mean(np.linalg.norm(frac_ps - all_ps[tmp_indices], axis=1))
            if dist < min_dist:
                index = j
                min_dist = dist
        tmp_idx = trees[index].query_ball_point(frac_ps, radius, workers=-1)
        for k in range(len(tmp_idx)):
            indices = np.concatenate([indices, np.array(tmp_idx[k]).reshape(-1, 1)], axis=0)
        indices = indices.astype(np.int).flatten()
        mask = np.ones(all_points[index].shape, dtype=bool)
        mask[indices, :] = False
        rest_points[index] = all_points[index][mask].reshape(-1, 3)
    rest_pcds = []
    print("rest_points", np.array(rest_points).shape)
    print("rest_points", np.array(rest_points))
    for i in range(len(rest_points)):
        pcd = o3d.geometry.PointCloud()
        print(rest_points[i])
        pcd.points = o3d.utility.Vector3dVector(rest_points[i])
        rest_pcds.append(pcd)
    return rest_pcds
