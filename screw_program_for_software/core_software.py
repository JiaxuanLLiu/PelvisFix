# coding = utf-8
import screw_program_for_software.geometry_software as geometry_software
import numpy as np
import open3d as o3d
import scipy.spatial as spatial
import scipy
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import screw_program_for_software.screw_setting as screw_setting
import time


def get_screw_dir_by_SVM(points1, points2, svm_threshold=screw_setting.svm_threshold):
    plane_normal, _ = geometry_software.fit_plane_by_svm(points1, points2, svm_threshold)
    return plane_normal


def get_screw_dir_by_Center(points1, points2):
    center1 = np.mean(points1, axis=0)
    center2 = np.mean(points2, axis=0)
    normal = center1 - center2
    normal = normal/np.linalg.norm(normal)
    return normal


def get_screw_dir_by_ransac(points1, points2):
    points = np.concatenate([points1, points2], axis=0)
    plane_info, _, _ = geometry_software.ransac_planefit(points, ransac_n=3, max_dst=screw_setting.ransac_eps)
    return plane_info[3:6]


# Chebyshev center 
def get_screw_implant_position_by_Chebyshev_center(points1, points2):
    points1 = np.array(points1)
    points2 = np.array(points2)
    all_points = np.concatenate([points1, points2], axis=0)
    pca = PCA()
    pca.fit(all_points)
    vec1 = np.array(pca.components_[0, :])
    vec1 = vec1/np.linalg.norm(vec1)
    dsp_dsbt = np.dot(all_points, vec1)
    min_val = dsp_dsbt[np.argmin(dsp_dsbt)]
    max_val = dsp_dsbt[np.argmax(dsp_dsbt)]
    res = max_val - min_val
    s_r = get_screw_radius()
    initial_center = np.mean(all_points, axis=0)
    normal = get_screw_dir_by_SVM(points1, points2)
    if res < 12*s_r:
        return initial_center
    else:
        tmp_position = min_val + 6*s_r
        max_res = 0
        best_center = initial_center
        last_num = 0
        while tmp_position <= max_val - max(6*s_r, 0.2*res):
            indices = np.array(np.where((dsp_dsbt > tmp_position - 6*s_r) & (dsp_dsbt < tmp_position + 6*s_r))).flatten()
            if indices.shape[0] <= 60:
                tmp_position = tmp_position + s_r
                continue
            tmp_points = all_points[indices]
            normal1 = np.array([normal[2], 0, -normal[0]])
            normal1 = normal1/np.linalg.norm(normal1)
            normal2 = np.cross(normal1, normal)
            normal3 = normal1 - normal2
            normal3 = np.linalg.norm(normal3)
            normal4 = normal1 + normal2
            normal4 = normal4/np.linalg.norm(normal4)
            normals = [normal1, normal2, normal3, normal4]
            center = np.mean(tmp_points, axis=0)
            init_length = 0.2
            tmp_res = 0
            while tmp_res == 0:
                for i in range(4):
                    tmp_center = center + init_length*normals[i]
                    tmp_norm = np.linalg.norm(all_points - np.expand_dims(tmp_center, 0).repeat(all_points.shape[0], axis=0), axis=1)
                    indices = np.array(np.where(tmp_norm < 3)).flatten()
                    if indices.shape[0] == 0:
                        tmp_res = init_length
                        break
                    tmp_center = center - init_length*normals[i]
                    tmp_norm = np.linalg.norm(all_points - np.expand_dims(tmp_center, 0).repeat(all_points.shape[0], axis=0), axis=1)
                    indices = np.array(np.where(tmp_norm < 3)).flatten()
                    if indices.shape[0] == 0:
                        tmp_res = init_length
                        break
                init_length = init_length + 0.2

            tmp_position = tmp_position + s_r
            if tmp_res > max_res or (tmp_res == max_res and indices.shape[0] > last_num):
                max_res = tmp_res
                last_num = indices.shape[0]
                best_center = np.mean(tmp_points, axis=0)
        return best_center


def get_2_screw_implant_positions_by_Chebyshev_center(points1, points2, length_rate):
    points1 = np.array(points1)
    points2 = np.array(points2)
    all_points = np.concatenate([points1, points2], axis=0)
    pca = PCA()
    pca.fit(all_points)
    vec1 = np.array(pca.components_[0, :])
    vec1 = vec1/np.linalg.norm(vec1)
    dsp_dsbt = np.dot(all_points, vec1)
    min_val = dsp_dsbt[np.argmin(dsp_dsbt)]
    max_val = dsp_dsbt[np.argmax(dsp_dsbt)]
    s_r = get_screw_radius()
    tmp_position = min_val + 6*s_r
    max_res1 = 0
    max_res2 = 0
    best_center1 = np.mean(points1, axis=0)
    best_center2 = np.mean(points2, axis=0)
    normal = geometry_software.ransac_planefit(all_points, 3, max_dst=2*screw_setting.ransac_eps)[0][3:6]
    while tmp_position <= max_val - max(6*s_r, 0.2*(max_val - min_val)):
        indices = np.array(np.where((dsp_dsbt > tmp_position - 6*s_r) & (dsp_dsbt < tmp_position + 6*s_r))).flatten()
        if indices.shape[0] <= 60:
            tmp_position = tmp_position + s_r
            continue
        tmp_points = all_points[indices]
        normal1 = np.array([normal[2], 0, -normal[0]])
        normal1 = normal1/np.linalg.norm(normal1)
        normal2 = np.cross(normal1, normal)
        normal3 = normal1 - normal2
        normal3 = np.linalg.norm(normal3)
        normal4 = normal1 + normal2
        normal4 = normal4/np.linalg.norm(normal4)
        normals = [normal1, normal2, normal3, normal4]
        center = np.mean(tmp_points, axis=0)
        init_length = 0.2
        tmp_res = 0
        while tmp_res == 0:
            for i in range(4):
                tmp_center = center + init_length*normals[i]
                tmp_norm = np.linalg.norm(all_points - np.expand_dims(tmp_center, 0).repeat(all_points.shape[0], axis=0), axis=1)
                indices = np.array(np.where(tmp_norm < 3)).flatten()
                if indices.shape[0] == 0:
                    tmp_res = init_length
                    break
                tmp_center = center - init_length*normals[i]
                tmp_norm = np.linalg.norm(all_points - np.expand_dims(tmp_center, 0).repeat(all_points.shape[0], axis=0), axis=1)
                indices = np.array(np.where(tmp_norm < 3)).flatten()
                if indices.shape[0] == 0:
                    tmp_res = init_length
                    break
            init_length = init_length + 0.2
        tmp_position = tmp_position + s_r
        if tmp_res >= max_res1:
            if np.linalg.norm(center - best_center1) > length_rate*s_r:
                max_res2 = max_res1
                best_center2 = best_center1
                max_res1 = tmp_res
                best_center1 = center
            elif np.linalg.norm(center - best_center2) > length_rate*s_r:
                max_res1 = tmp_res
                best_center1 = center
        elif tmp_res >= max_res2 and np.linalg.norm(center - best_center1) > length_rate*s_r:
            max_res2 = tmp_res
            best_center2 = center
    return best_center1, best_center2


def get_screw_radius(radius=screw_setting.screw_radius):
    return radius


def separate_point(points, radius=screw_setting.sp_radius, eps1=screw_setting.sp_threshold):
    points = np.array(points)
    all_points = []
    y_pred = DBSCAN(eps=eps1).fit_predict(points)
    y_uniq = np.unique(np.array(y_pred))
    for y in y_uniq:
        indices = np.argwhere(y_pred == y).flatten()
        ps = np.array(points[indices])
        if ps.shape[0] / points.shape[0] < 0.1 and ps.shape[0] < 10:
            continue
        all_points.append(ps)
    return all_points


def get_effect_points(pcds, threshold=screw_setting.gep_threshold):
    all_points = [0]
    tmp_all_points = []
    for pcd in pcds:
        points = np.array(pcd.points)
        tmp_all_points.append(points)
    all_points[0] = tmp_all_points[0].copy()
    for i in range(1, len(tmp_all_points)):
        for j in range(len(all_points)):
            if tmp_all_points[i].shape[0] <= all_points[j].shape[0]:
                all_points.insert(j, tmp_all_points[i])
                break
            elif j == len(all_points) - 1:
                all_points.append(tmp_all_points[i])
                break
    trees = []
    for points in all_points:
        tree = spatial.KDTree(points)
        trees.append(tree)
    finish_indices = []
    match_clusters = []
    for i in range(0, len(all_points)):
        finish_indices.append(i)
        points1 = np.empty((0, 3))
        points2 = np.empty((0, 3))
        for j in range(0, len(trees)):
            if j not in finish_indices:
                _, indices = trees[j].query(all_points[i], 1, workers=-1)
                indices = np.array(indices)
                dist = np.linalg.norm(all_points[i] - all_points[j][indices], axis=1)
                dist_idx = np.argwhere(dist < threshold).flatten()
                points1 = np.concatenate([points1, all_points[i][dist_idx]], axis=0)
                points2 = np.concatenate([points2, all_points[j][indices[dist_idx]]], axis=0)
        if points1.shape[0] > 50:
            match_clusters.append([points1, points2])
    refine_cluster = []
    for match_cluster in match_clusters:
        points1 = match_cluster[0]
        points2 = match_cluster[1]
        all_points = separate_point(points1)
        tree = spatial.KDTree(points2)
        for points in all_points:
            _, indices = tree.query(points, 1, workers=-1)
            refine_cluster.append([points, points2[indices]])
    return refine_cluster


def get_cone(dire, angle=screw_setting.cone_angle, r_resolution=screw_setting.r_res, c_resolution=screw_setting.c_res):
    orth_dir = np.array([dire[2], 0, -dire[0]])
    orth_dir = orth_dir/np.linalg.norm(orth_dir)
    radius = np.tan(angle)
    rot_mtx = scipy.linalg.expm(np.cross(np.eye(3), dire/scipy.linalg.norm(dire)*2*np.pi/c_resolution))
    cone = [dire]
    for i in range(c_resolution):
        orth_dir = np.dot(rot_mtx, orth_dir)
        for j in range(r_resolution):
            n_dir = dire + orth_dir*radius*j/r_resolution
            n_dir = n_dir/np.linalg.norm(n_dir)
            cone.append(n_dir)
    return cone


def estimate_dist(info1, info2):
    dire1 = info1[0]
    cent1 = info1[1]
    f_length1 = info1[4]
    b_length1 = info1[5]
    f_p1 = cent1 + dire1*f_length1
    b_p1 = cent1 - dire1*b_length1

    dire2 = info2[0]
    cent2 = info2[1]
    f_length2 = info2[4]
    b_length2 = info2[5]
    f_p2 = cent2 + dire2*f_length2
    b_p2 = cent2 - dire2*b_length2
    return geometry_software.segment_3d_dist(f_p1, b_p1, f_p2, b_p2)


def isInterference(new_info, path_infos, eps=2.5*screw_setting.screw_radius):
    for info in path_infos:
        dist = estimate_dist(new_info, info)
        if dist <= eps:
            return True, info
    return False, None


def isExploreV1(pcds, info, radius=2*screw_setting.screw_radius, rate=1.2):
    length1 = info[4]
    all_points = np.empty((0, 3))
    for pcd in pcds:
        all_points = np.concatenate([all_points, np.array(pcd.points)], axis=0)
    dire = np.array(info[0])
    dire = dire/np.linalg.norm(dire)
    cent = np.array(info[1])
    dire1 = np.array([dire[2], 0, -dire[0]])
    if np.linalg.norm(dire1) == 0:
        dire1 = np.array([0, -dire[2], dire[1]])
    dire1 = dire1/np.linalg.norm(dire1)
    dire2 = np.cross(dire, dire1)
    dire2 = dire2/np.linalg.norm(dire2)
    dire3 = dire1 - dire2
    dire3 = dire3/np.linalg.norm(dire3)
    dire4 = dire1 + dire2
    dire4 = dire4/np.linalg.norm(dire4)
    dires = [dire1, dire2, dire3, dire4]
    test_point1 = cent + length1*dire
    diff = all_points - np.expand_dims(test_point1, 0).repeat(all_points.shape[0], axis=0)
    diff_norm = np.linalg.norm(diff, axis=1)
    for i in range(0, len(dires)):
        r_dist = np.sqrt(diff_norm**2 - np.abs(np.dot(diff, dires[i].T))**2)
        indices = np.argwhere(r_dist < radius).flatten()
        if indices.shape[0] <= 1:
            return True
        flag1 = False
        flag2 = False
        dists = np.dot(diff[indices], dires[i].T)
        for dist in dists:
            if not flag1:
                if dist < 200 and dist > rate*get_screw_radius():
                    flag1 = True
            if not flag2:
                if dist > -200 and dist < -rate*get_screw_radius():
                    flag2 = True
            if flag1 and flag2:
                break
        if not flag1 or not flag2:
            return True
    return False


def isExploreV2(pcds, info, radius=2*screw_setting.screw_radius, rate=1.2):
    length2 = info[5]
    all_points = np.empty((0, 3))
    for pcd in pcds:
        all_points = np.concatenate([all_points, np.array(pcd.points)], axis=0)
    dire = np.array(info[0])
    cent = np.array(info[1])
    dire1 = np.array([dire[2], 0, -dire[0]])
    if np.linalg.norm(dire1) == 0:
        dire1 = np.array([0, -dire[2], dire[1]])
    dire1 = dire1/np.linalg.norm(dire1)
    dire2 = np.cross(dire, dire1)
    dire2 = dire2/np.linalg.norm(dire2)
    dire3 = dire1 - dire2
    dire3 = dire3/np.linalg.norm(dire3)
    dire4 = dire1 + dire2
    dire4 = dire4/np.linalg.norm(dire4)
    dires = [dire1, dire2, dire3, dire4]
    test_point2 = cent - (length2)*dire
    diff = all_points - np.expand_dims(test_point2, 0).repeat(all_points.shape[0], axis=0)
    diff_norm = np.linalg.norm(diff, axis=1)
    cone_pcd = []
    for i in range(0, len(dires)):
        r_dist = np.sqrt(diff_norm**2 - np.abs(np.dot(diff, dires[i].T))**2)
        indices = np.argwhere(r_dist < radius).flatten()
        tmp_points = all_points[indices]
        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = o3d.utility.Vector3dVector(tmp_points)
        cone_pcd.append(tmp_pcd)
        if indices.shape[0] <= 1:
            return True
        flag1 = False
        flag2 = False
        dists = np.dot(diff[indices], dires[i].T)
        for dist in dists:
            if not flag1:
                if dist < 200 and dist > rate*get_screw_radius():
                    flag1 = True
            if not flag2:
                if dist > -200 and dist < -rate*get_screw_radius():
                    flag2 = True
            if flag1 and flag2:
                break
        if not flag1 or not flag2:
            return True
    return False


def get_optimal_info(path_info, rest_pcds, rest_pcds_for_explore, eps=screw_setting.screw_radius, dist_eps=screw_setting.dist_eps):
    rf_path_info = []
    rest_points = []
    restPoints = np.empty((0, 3))
    for pcd in rest_pcds:
        points = np.asarray(pcd.points)
        restPoints = np.concatenate([restPoints, points], axis=0)
        rest_points.append(points)
    allCenter = np.mean(restPoints, axis=0)
    for i in range(len(path_info)):
        start = time.time()
        info = path_info[i]
        dire = info[0]
        cent = info[1]
        id1 = info[2]
        id2 = info[3]
        cone = get_cone(dire)
        best_dir = dire
        best_length1 = 0
        best_length2 = 0
        close_length2 = 0
        cent_var = restPoints - np.expand_dims(cent, 0).repeat(restPoints.shape[0], axis=0)
        norm = np.linalg.norm(cent_var, axis=1)
        best_cone_pcd = []
        for j in range(len(cone)): # tqdm(range(len(cone)), desc="\033[31mThe %dth screw:\033[0m" % (i + 1),):
            n_dir = cone[j]
            r_dist = np.sqrt(norm**2 - np.abs(np.dot(cent_var, n_dir.T))**2)
            indices = np.argwhere(r_dist < eps).flatten()
            if indices.shape[0] == 0:
                continue
            tmp_points = restPoints[indices]
            y_pred = DBSCAN(eps=dist_eps/2).fit_predict(tmp_points)
            y_uniq = np.unique(np.array(y_pred))
            fp_list = []
            cp_list = []
            ps = None
            cone_pcd = []
            for y in y_uniq:
                idx = np.argwhere(y_pred == y).flatten()
                ps = np.array(tmp_points[idx])
                dist = np.dot(ps, n_dir.T)
                cp_list.append(ps[np.argmin(dist).flatten()[0]])
                fp_list.append(ps[np.argmax(dist).flatten()[0]])
            fp_list = np.array(fp_list)
            cp_list = np.array(cp_list)
            ori_cent = np.expand_dims(cent, 0).repeat(fp_list.shape[0], axis=0)
            dir_fp = fp_list - ori_cent
            dir_cp = cp_list - ori_cent
            com_fp = np.linalg.norm(dir_fp, axis=1)
            com_cp = np.linalg.norm(dir_cp, axis=1)
            length1 = 0
            length2 = 0
            tmp_length1 = 0
            tmp_length2 = 0
            idx1 = -1
            idx2 = -1
            if com_fp.shape[0] <= 1:
                continue
            for k in range(com_fp.shape[0]):
                if com_fp[k] < com_cp[k]:
                    tmp_com = com_cp[k].copy()
                    com_cp[k] = com_fp[k]
                    com_fp[k] = tmp_com
                    tmp_diff = dir_cp[k].copy()
                    dir_cp[k] = dir_fp[k]
                    dir_fp[k] = tmp_diff
            tmp_com_cp = np.array([com_cp[0]])
            tmp_com_fp = np.array([com_fp[0]])
            tmp_dir_cp = np.array([dir_cp[0]])
            for m in range(1, com_cp.shape[0]):
                for k in range(tmp_com_cp.shape[0]):
                    if com_cp[m] <= tmp_com_cp[k]:
                        tmp_com_cp = np.insert(tmp_com_cp, k, com_cp[m], axis=0)
                        tmp_com_fp = np.insert(tmp_com_fp, k, com_fp[m], axis=0)
                        tmp_dir_cp = np.insert(tmp_dir_cp, k, dir_cp[m], axis=0)
                        break
                    elif k == tmp_com_cp.shape[0] - 1:
                        tmp_com_cp = np.concatenate([tmp_com_cp, [com_cp[m]]], axis=0)
                        tmp_com_fp = np.concatenate([tmp_com_fp, [com_fp[m]]], axis=0)
                        tmp_dir_cp = np.concatenate([tmp_dir_cp, [dir_cp[m]]], axis=0)
                        break
            com_cp = tmp_com_cp
            com_fp = tmp_com_fp
            dir_cp = tmp_dir_cp
            explore_flag1 = False
            explore_flag2 = False
            for k in range(com_fp.shape[0]):
                pn = np.dot(dir_cp[k], n_dir.T)
                if pn > 0 and not explore_flag1:
                    tmp_length1 = com_cp[k]
                    if idx1 == -1:
                        explore_length1 = tmp_length1/2
                        if not isExploreV1(rest_pcds_for_explore, [n_dir, cent, id1, id2, explore_length1, 0]):
                            length1 = tmp_length1
                            idx1 = k
                        else:
                            explore_flag1 = True
                    else:
                        explore_length1 = (tmp_length1 + com_fp[idx1])/2
                        if not isExploreV1(rest_pcds_for_explore, [n_dir, cent, id1, id2, explore_length1, 0]) and not isExploreV1(rest_pcds_for_explore, [n_dir, cent, id1, id2, (com_cp[idx1] + com_fp[idx1])/2, 0]):
                            length1 = tmp_length1
                            idx1 = k
                        else:
                            explore_flag1 = True
                elif pn <= 0 and not explore_flag2:
                    tmp_length2 = com_cp[k]
                    if idx2 == -1:
                        explore_length2 = tmp_length2/2
                        if not isExploreV2(rest_pcds_for_explore, [n_dir, cent, id1, id2, 0, explore_length2]):
                            length2 = tmp_length2
                            idx2 = k
                        else:
                            explore_flag2 = True
                    else:
                        explore_length2 = (tmp_length2 + com_fp[idx2])/2
                        if not isExploreV2(rest_pcds_for_explore, [n_dir, cent, id1, id2, 0, explore_length2]) and not isExploreV2(rest_pcds_for_explore, [n_dir, cent, id1, id2, 0, (com_cp[idx2] + com_fp[idx2])/2]):
                            length2 = tmp_length2
                            idx2 = k
                        else:
                            explore_flag2 = True
            if idx1 == -1 or idx2 == -1 or length1 <= 3*dist_eps or length2 <= 3*dist_eps:
                continue
            if np.linalg.norm(cent + n_dir*length1 - allCenter) > np.linalg.norm(cent - n_dir*length2 - allCenter):
                n_dir = - n_dir
                tmp_idx = idx1
                idx1 = idx2
                idx2 = tmp_idx
                length1 = com_cp[idx1]
                length2 = min(com_fp[idx2], com_cp[idx2] + 4)
            interference_info = rf_path_info.copy()
            for k in range(len(rf_path_info) + 1, len(path_info)):
                if k != i:
                    interference_info.append([path_info[k][0], path_info[k][1], path_info[k][2], path_info[k][3], dist_eps, dist_eps])
            ret, _ = isInterference([n_dir, cent, id1, id2, length1, length2], interference_info)
            new_length1 = length1
            new_length2 = length2
            flag_interference1 = True
            ret1 = ret
            ret2 = ret
            if ret and isInterference([n_dir, cent, id1, id2, 3*dist_eps, 3*dist_eps], interference_info)[0]:
                continue
            while ret1 or ret2:
                if flag_interference1:
                    new_length1 = new_length1*0.9
                    if new_length1 < max(best_length1, 3*dist_eps):
                        new_length1 = 0
                        break
                    com_cp[idx1] = new_length1
                    com_fp[idx1] = new_length1
                    ret1, _ = isInterference([n_dir, cent, id1, id2, new_length1, 3*dist_eps], interference_info)
                    if not ret1:
                        flag_interference1 = False
                else:
                    new_length2 = new_length2*0.9
                    if new_length2 < max(best_length2, 3*dist_eps):
                        new_length2 = 0
                        break
                    com_cp[idx2] = new_length2
                    com_fp[idx2] = new_length2
                    ret2, _ = isInterference([n_dir, cent, id1, id2, new_length1, new_length2], interference_info)
            length1 = new_length1
            length2 = new_length2 
            if np.abs(length1) + np.abs(length2) < 8*dist_eps or min(length1, length2) < 3*dist_eps or (min(length1, length2)/max(length1, length2) < 0.33 and min(length1, length2) < 20):
                continue
            continue_ornot = True
            if com_cp[idx1] + com_cp[idx2] > best_length1 + close_length2:
                continue_ornot = False
            if not continue_ornot:
                best_length1 = length1
                best_length2 = length2
                close_length2 = com_cp[idx2]
                best_dir = n_dir
        end = time.time()
        if best_length1 == 0 or best_length2 == 0:
            continue
        print("Screw %d direction planning time:%.2fs, length1:%.2f, length2:%.2f" % (len(rf_path_info)+1, end-start, best_length1, best_length2))
        rf_path_info.append([best_dir, cent, id1, id2, best_length1, best_length2])
    return rf_path_info


def initial_program(frac_pcds, all_pcds, rest_pcds):
    start = time.time()
    refine_cluster = get_effect_points(frac_pcds)
    path_info = []
    all_points = []
    frac_points = []
    sizes = [0]
    id_record = []
    frac_id = []
    for pcd in frac_pcds:
        frac_points.append(np.asarray(pcd.points))
    for i in range(len(all_pcds)):
        pcd = all_pcds[i]
        all_points.append(np.asarray(pcd.points))
        sizes.append(sizes[i] + np.asarray(pcd.points).shape[0])
        id_record.append(0)
    for i in range(len(refine_cluster)): #tqdm(range(len(refine_cluster)), desc="\033[31mInitializing implantation centers\033[0m"):
        points = refine_cluster[i]
        points1 = points[0]
        points2 = points[1]
        path_dir = get_screw_dir_by_SVM(points1, points2)
        path_center = get_screw_implant_position_by_Chebyshev_center(points1, points2)
        tmp_p1 = points1[1, :]
        tmp_p2 = points2[1, :]
        id1 = None
        id2 = None
        f_id1 = None
        f_id2 = None
        for j in range(len(frac_points)):
            t1 = np.sum(np.abs(frac_points[j] - np.expand_dims(tmp_p1, 0).repeat(frac_points[j].shape[0], axis=0)), axis=1)
            t2 = np.sum(np.abs(frac_points[j] - np.expand_dims(tmp_p2, 0).repeat(frac_points[j].shape[0], axis=0)), axis=1)
            if np.where(t1 == 0)[0].shape[0] != 0:
                f_id1 = j
            if np.where(t2 == 0)[0].shape[0] != 0:
                f_id2 = j
        frac_id.append([f_id1, f_id2])
        
        allPoints1 = np.empty((0, 3))
        for j in range(len(all_points)):
            allPoints1 = np.concatenate([allPoints1, all_points[j]], axis=0)
        t1 = np.sum(np.abs(allPoints1 - np.expand_dims(tmp_p1, 0).repeat(allPoints1.shape[0], axis=0)), axis=1)
        index1 = np.argmin(t1).flatten()[0]
        for j in range(len(sizes)):
            if index1 == 0:
                id1 = 0
                break
            elif index1 < sizes[j]:
                id1 = j - 1
                break
        allPoints2 = np.empty((0, 3))
        for j in range(len(all_points)):
            if j != id1:
                allPoints2 = np.concatenate([allPoints2, all_points[j]], axis=0)
            else:
                allPoints2 = np.concatenate([allPoints2, np.ones([all_points[j].shape[0], 3])*10000], axis=0)
        t2 = np.linalg.norm(allPoints2 - np.expand_dims(tmp_p2, 0).repeat(allPoints2.shape[0], axis=0), axis=1)
        index2 = np.argmin(t2).flatten()[0]
        for j in range(len(sizes)):
            if index2 == 0:
                if id1 == 0:
                    id2 = 1
                else:
                    id2 = 0
                break
            elif index2 < sizes[j]:
                id2 = j - 1
                break
        path_info.append([path_dir, path_center, id1, id2, 0, 0])
    # refine the number of implanted screws
    for info in path_info:
        id_record[info[2]] = id_record[info[2]] + 1
        id_record[info[3]] = id_record[info[3]] + 1
    for i in range(len(id_record)):
        if id_record[i] < 2:
            for j in range(len(path_info)):
                info = path_info[j]
                if info[2] == i or info[3] == i:
                    frac_idx = None
                    for k in range(len(frac_id)):
                        if frac_id[k][0] == i or frac_id[k][1] == i:
                            frac_idx = k
                            break
                    point1 = refine_cluster[frac_idx][0]
                    point2 = refine_cluster[frac_idx][1]
                    points = np.concatenate([point1, point2], axis=0)
                    pca = PCA()
                    pca.fit(points)
                    vec = np.array(pca.components_[0, :])
                    vec = vec/np.linalg.norm(vec)
                    
                    dsp_dsbt = np.dot(points, vec.T)
                    min_val = dsp_dsbt[np.argmin(dsp_dsbt)]
                    max_val = dsp_dsbt[np.argmax(dsp_dsbt)]
                    res = max_val - min_val
                    length_rate = 24
                    if res > length_rate*get_screw_radius():
                        path_center1, path_center2 = get_2_screw_implant_positions_by_Chebyshev_center(point1, point2, length_rate/6)
                        info1 = [info[0], path_center1, info[2], info[3], 0, 0]
                        info2 = [info[0], path_center2, info[2], info[3], 0, 0]
                        id_record[info[2]] = id_record[info[2]] + 1
                        id_record[info[3]] = id_record[info[3]] + 1
                        path_info[j] = info1
                        j = j + 1
                        path_info.insert(j, info2)
                        break
    path_info = refine_path_info(path_info, rest_pcds)
    end = time.time()
    print("time:%.2fs"%(end-start))
    return path_info


def refine_path_info(path_info, pcds, radius=screw_setting.path_refine_radius, length_eps=screw_setting.length_eps):
    rf_path_info = []
    all_points = np.empty((0, 3))
    allPoints = []
    for pcd in pcds:
        points = np.asarray(pcd.points)
        all_points = np.concatenate([all_points, points], axis=0)
        allPoints.append(points)
    tree = spatial.KDTree(all_points)
    centers = []
    for i in range(len(path_info)):
        info = path_info[i]
        point = info[1]
        centers.append(point)
        direc = info[0]
        id1 = info[2]
        id2 = info[3]
        indices = tree.query_ball_point(point, radius, workers=-1)
        indices = np.array(indices).flatten()
        points = all_points[indices]
        pca = PCA()
        pca.fit(points)
        vec = np.array(pca.components_[0, :])
        vec = vec/np.linalg.norm(vec)
        if np.dot(vec, direc.T) < 0:
            vec = -vec
        rf_direc = vec  #+ path_info[i][0]
        rf_path_info.append([rf_direc, point, id1, id2, 0, 0])
    for i in range(len(path_info)):
        dire = np.mean(allPoints[path_info[i][2]], axis=0) - np.mean(allPoints[path_info[i][3]], axis=0)
        dire = dire/np.linalg.norm(dire)
        if np.abs(np.dot(dire, vec.T)) < 0.8:
            rf_path_info[i][0] = dire
    return rf_path_info
