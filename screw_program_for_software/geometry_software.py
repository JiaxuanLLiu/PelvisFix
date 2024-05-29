# coding = utf-8
import numpy as np
from scipy.optimize import leastsq
from sklearn import svm
import random


def fit_func(p, x, y):
    """ 数据拟合函数 """
    a, b, c = p
    return a * x + b * y + c


def residuals(p, x, y, z):
    """ 误差函数 """
    return z - fit_func(p, x, y)


def estimate_plane_with_leastsq(pts):
    """ 根据最小二乘拟合出平面参数 """
    p0 = [1, 0, 1]
    np_pts = np.array(pts)
    plsq = leastsq(residuals,
                   p0,
                   args=(np_pts[:, 0], np_pts[:, 1], np_pts[:, 2]))
    print(plsq)
    return plsq[0]


def get_proper_plane_params(p, pts):
    """ 根据拟合的平面的参数，得到实际显示的最佳的平面 """
    np_pts = np.array(pts)

    np_pts_mean = np.mean(np_pts, axis=0)
    np_pts_min = np.min(np_pts, axis=0)
    np_pts_max = np.max(np_pts, axis=0)

    plane_center_z = fit_func(p, np_pts_mean[0], np_pts_mean[1])
    plane_center = np.array([np_pts_mean[0], np_pts_mean[1], plane_center_z])

    plane_origin_z = fit_func(p, np_pts_min[0], np_pts_min[1])
    plane_origin = np.array([np_pts_min[0], np_pts_min[1], plane_origin_z])

    if np.linalg.norm(p) < 1e-10:
        print(r'plsq 的 norm 值为 0 {}'.format(p))
    plane_normal = p / np.linalg.norm(p)

    plane_pt_1 = np.array([
        np_pts_max[0], np_pts_min[1],
        fit_func(p, np_pts_max[0], np_pts_min[1])
    ])
    plane_pt_2 = np.array([
        np_pts_min[0], np_pts_max[1],
        fit_func(p, np_pts_min[0], np_pts_max[1])
    ])
    return plane_center, plane_normal, plane_origin, plane_pt_1, plane_pt_2


def fit_plane_by_norm2(points):
    p = estimate_plane_with_leastsq(points)
    return get_proper_plane_params(p, points)


def fit_plane_by_svm(points1, points2, threshold):
    cls = svm.SVC(kernel="linear", C=threshold)
    labels1 = np.ones(points1.shape[0])
    labels2 = -np.ones(points2.shape[0])
    X = np.concatenate([points1, points2], axis=0)
    Y = np.concatenate([labels1, labels2], axis=0)
    cls.fit(X, Y)
    w = cls.coef_[0]
    b = cls.intercept_[0]
    sv = cls.support_vectors_[-1]
    z = -(w[0] * sv[0] + w[1] * sv[1] + b) / w[2]
    plane_center = [sv[0], sv[1], z]
    w = w / np.linalg.norm(w)
    return w, plane_center


def SVD(points):
    # 二维，三维均适用
    # 二维直线，三维平面
    pts = points.copy()
    # 奇异值分解
    c = np.mean(pts, axis=0)
    A = pts - c  # shift the points
    A = A.T  # 3*n
    u, s, vh = np.linalg.svd(A, full_matrices=False,
                             compute_uv=True)  # A=u*s*vh
    normal = u[:, -1]

    # 法向量归一化
    nlen = np.sqrt(np.dot(normal, normal.T))
    normal = normal / nlen
    # normal 是主方向的方向向量 与PCA最小特征值对应的特征向量是垂直关系
    # u 每一列是一个方向
    # s 是对应的特征值
    # c >>> 点的中心
    # normal >>> 拟合的方向向量
    return u, s, c, normal


class plane_model(object):

    def __init__(self):
        self.parameters = None

    def calc_inliers(self, points, dst_threshold):
        c = self.parameters[0:3]
        n = self.parameters[3:6]
        dst = abs(np.dot(points - c, n.T))
        ind = dst < dst_threshold
        return ind

    def estimate_parameters(self, pts):
        num = pts.shape[0]
        if num == 3:
            c = np.mean(pts, axis=0)
            l1 = pts[1] - pts[0]
            l2 = pts[2] - pts[0]
            n = np.cross(l1, l2)    
            n = n / np.linalg.norm(n)
        else:
            _, _, c, n = SVD(pts)

        params = np.hstack((c.reshape(1, -1), n.reshape(1, -1)))[0, :]
        self.parameters = params
        return params

    def set_parameters(self, parameters):
        self.parameters = parameters


def ransac_planefit(points,
                    ransac_n,
                    max_dst,
                    max_trials=500,
                    stop_inliers_ratio=1.0,
                    initial_inliers=None):
    # RANSAC 平面拟合
    pts = points.copy()
    num = pts.shape[0]
    # cc = np.mean(pts, axis=0)
    iter_max = max_trials
    best_inliers_ratio = 0  # 符合拟合模型的数据的比例
    best_plane_params = None
    # best_inliers = None
    # best_remains = None
    for i in range(iter_max):
        sample_points = None
        while 1:
            sample_index = random.sample(range(num), ransac_n)
            sample_points = pts[sample_index, :]
            l1 = sample_points[1] - sample_points[0]
            l2 = sample_points[2] - sample_points[0]
            n = np.cross(l1, l2)    
            if np.linalg.norm(n) != 0:
                break
        plane = plane_model()
        plane_params = plane.estimate_parameters(sample_points)
        #  计算内点 外点
        index = plane.calc_inliers(points, max_dst)
        inliers_ratio = pts[index].shape[0] / num

        if inliers_ratio > best_inliers_ratio:
            best_inliers_ratio = inliers_ratio
            best_plane_params = plane_params
            bset_inliers = pts[index]
            bset_remains = pts[index == False]

        if best_inliers_ratio > stop_inliers_ratio:
            # # 检查是否达到最大的比例
            # print("iter: %d\n" % i)
            # print("best_inliers_ratio: %f\n" % best_inliers_ratio)
            break
    return best_plane_params, bset_inliers, bset_remains


def distace(point1, point2):
    return np.linalg.norm(point1 - point2)


def distace_3d(p1, p2, q1, q2):
    dire1 = p1 - p2
    dire2 = q1 - q2
    dire1 = dire1/np.linalg.norm(dire1)
    dire2 = dire2/np.linalg.norm(dire2)
    normal = np.cross(dire1, dire2)
    project_dif = np.abs(np.dot(normal, p1.T) - np.dot(normal, q1.T))
    return project_dif


def line_3d_relationship(p1, p2, q1, q2):
    v1 = p1 - p2
    v2 = q1 - q2
    if np.linalg.norm(v1) == 0 and np.linalg.norm(v2) == 0:
        return np.linalg.norm(p1 - q1), p1, q1
    elif np.linalg.norm(v1) == 0:
        if np.dot(p1 - q1, (p1 - q2).T)/np.linalg.norm(p1 - q1)/np.linalg.norm(p1 - q2) == -1:
            return 0, p1, p1
        elif np.dot(p1 - q1, (p1 - q2).T)/np.linalg.norm(p1 - q1)/np.linalg.norm(p1 - q2) == 1:
            return min(np.linalg.norm(p1 - q1), np.linalg.norm(p1 - q2)), p1, p1
        else:
            v1 = p1 - q1
            normal = np.cross(v1, v2)
            normal = normal/np.linalg.norm(normal)
            cross_v2 = np.cross(v2, normal)
            cross_v2 = cross_v2/np.linalg.norm(cross_v2)
            t = np.dot(cross_v2, v1.T)
            cross_q = p1 - cross_v2*t
            return np.abs(t), p1, cross_q
    elif np.linalg.norm(v2) == 0:
        if np.dot(q1 - p1, (q1 - p2).T)/np.linalg.norm(q1 - p1)/np.linalg.norm(q1 - p2) == -1:
            return 0, q1, q1
        elif np.dot(q1 - p1, (q1 - p2).T)/np.linalg.norm(q1 - p1)/np.linalg.norm(q1 - p2) == 1:
            return min(np.linalg.norm(q1 - p1), np.linalg.norm(q1 - p2)), q1, q1
        else:
            v2 = q1 - p1
            normal = np.cross(v2, v1)
            normal = normal/np.linalg.norm(normal)
            cross_v1 = np.cross(v1, normal)
            cross_v1 = cross_v1/np.linalg.norm(cross_v1)
            t = np.dot(cross_v1, v2.T)
            cross_p = q1 - cross_v1*t
            return np.abs(t), cross_p, q1
    v1 = v1/np.linalg.norm(v1)
    v2 = v2/np.linalg.norm(v2)
    normal = np.cross(v1, v2)
    normal = normal/np.linalg.norm(normal)
    # test = np.abs(np.dot(normal, p1 - q1))
    cross_v1 = np.cross(v2, normal)
    cross_v1 = cross_v1/np.linalg.norm(cross_v1)
    cross_v2 = np.cross(v1, normal)
    cross_v2 = cross_v2/np.linalg.norm(cross_v2)
    cross_p = None
    cross_q = None
    t1 = None
    t2 = None
    # b = np.dot(v1, cross_v1.T)
    # c = np.dot(v2, cross_v2.T)
    if np.abs(np.dot(v1, cross_v1.T)) >= 0.01:
        t1 = np.dot(q1 - p1, cross_v1.T)/np.dot(v1, cross_v1.T)
    else:
        t1 = np.dot(q1 - p1, v1.T)/np.dot(v1, v1.T)
        t2 = 0
    if np.abs(np.dot(v2, cross_v2.T)) >= 0.01:
        t2 = np.dot(p1 - q1, cross_v2.T)/np.dot(v2, cross_v2.T)
    cross_p = p1 + t1*v1
    cross_q = q1 + t2*v2
    proj_dist = np.linalg.norm(cross_q - cross_p)
    return proj_dist, cross_p, cross_q


def segment_3d_dist(p1, p2, q1, q2):
    proj_dist, cross_p, cross_q = line_3d_relationship(p1, p2, q1, q2)
    if np.linalg.norm(p1 - p2) == 0 or np.linalg.norm(q1 - q2) == 0:
        return proj_dist
    p_inside = False
    if (cross_p[0] - p2[0])*(cross_p[0] - p1[0]) <= 0:
        p_inside = True
    q_inside = False
    if (cross_q[0] - q2[0])*(cross_q[0] - q1[0]) <= 0:
        q_inside = True
    l11 = np.linalg.norm(p1 - q1)
    l12 = np.linalg.norm(p1 - q2)
    l21 = np.linalg.norm(p2 - q1)
    l22 = np.linalg.norm(p2 - q2)
    if not p_inside and not q_inside:
        return min(min(l11, l12), min(l21, l22))
    elif p_inside and not q_inside:
        lp = np.linalg.norm(p1 - p2)
        return np.abs(min(np.sqrt(np.abs(l11**2 - ((lp**2 + l11**2 - l21**2)/lp/2)**2)),
                          np.sqrt(np.abs(l12**2 - ((lp**2 + l12**2 - l22**2)/lp/2)**2))))
    elif q_inside and not p_inside:
        lq = np.linalg.norm(q1 - q2)
        return np.abs(min(np.sqrt(np.abs(l12**2 - ((lq**2 + l12**2 - l11**2)/lq/2)**2)),
                          np.sqrt(np.abs(l22**2 - ((lq**2 + l22**2 - l21**2)/lq/2)**2))))
    return proj_dist
