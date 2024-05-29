# coding = utf-8
import os
import numpy as np

cwd = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')

data_name = 'data0092'

# the directory that store pcd files
pcd_dir = cwd + '/cases/' + data_name + '/face_pointcloud'

# the directory that store stl files
stl_dir = cwd + '/cases/' + data_name + '/stl_after_regis'

# stl outputpath
save_stl = cwd + '/export_screw_stl/' + data_name

# the directory that store npy files
# mtx_dir = cwd + '/data/data0014/trans_matrix'

# # the path of label file
# label_dir = cwd + '/data/label0014.nii.gz'

# # the path of image file
# img_dir = cwd + '/data/image0014.nii.gz'

# color space
color = [0,   1,   0,
         0,   0,   1,
         1,   1,   0,
         1,   0,   1,
         0,   1,   1,
         1,   1,   1]

# the number of neighborhoods in remove outliers
nd = 20

# the statistic ratio in remove outliers
std_rt = 2

# threhold in svm
svm_threshold = 2

# threshold in get_effect_pcd
gep_threshold = 3

# threhold in separate_point
sp_threshold = 6

# radius in separate_point
sp_radius = 60

# radius of screw
screw_radius = 1.5

# length of screw
screw_length = 30

# the ratio of line to screw in length
line_length_rate = 20

# radius in KDTree searching while refining path_info
path_refine_radius = 40

# eps in relu_refine_dir
rrd_eps_max = 0.4
rrd_eps_min = 0.3

# max distance in ransac
ransac_eps = 1

# angle eps in estimate screw length
angle_eps = 5

# distance eps in estimate screw length
dist_eps = 4

# cone angle in get_cone
cone_angle = np.pi/3

# radius resolution in get_cone
r_res = 15

# circle resolution in get_cone
c_res = 18  # 20°

# length threshold in refine_path_info
length_eps = 10*screw_radius

# point cloud down sample
voxel_size = 1

# resolution in explore
resolution = 8