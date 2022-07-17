'''
读入数据 保存生成的噪声图
'''
import importlib
from matplotlib import pyplot as plt
import open3d as o3d
import argparse
import logging
# import time
# import torch
from utils import exp_util
# -*- coding:utf-8 -*-
import numpy as np
from dataset.production import bilinear_interp, model_noise, shuffle_rand
import cv2
import copy, os
# from numba import jit
vis_param = argparse.Namespace()
vis_param.n_left_steps = 0
vis_param.args = None
vis_param.mesh_updated = True



if __name__ == '__main__':
    parser = exp_util.ArgumentParserX()
    args = parser.parse_args() #载入外部参数
    logging.basicConfig(level=logging.INFO)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    # Load in sequence. 数据接口
    seq_package, seq_class = args.sequence_type.split(".") #icl_nuim(data 的.py) ICLNUIMSequence 对应的接口类
    sequence_module = importlib.import_module("dataset.production." + seq_package) #dataset/production/icl_nuim.py
    sequence_module = getattr(sequence_module, seq_class) #class ICLNUIMSequence
    vis_param.sequence = sequence_module(**args.sequence_kwargs) #输入数据api class 运行init 默认不读入gt pose

    # 遍历每帧数据
    frame_reader = vis_param.sequence
    n_img = frame_reader.__len__()
    mydir = os.path.join(frame_reader.path, 'mydnoise')
    if not os.path.exists(mydir):
        os.mkdir(mydir)
        print('creat mydir at {}'.format(mydir))
    for idx in range(n_img):
        # frame_data = next(frame_reader) # 读入一一帧数据
        # if not (idx == frame_reader.frame_id - 1):
        #     raise AssertionError
        # if frame_reader.depth_noise:
        #     depth_noise = frame_data.noisedepth.cpu().numpy()
        #     depth_noise = depth_noise.astype(np.float32) # 单位m
        depth_img_path = str(frame_reader.path / frame_reader.depth_names[idx])
        # rgb_img_path = str(frame_reader.path / frame_reader.color_names[idx])
        # gt_c2w = frame_reader.gt_trajectory[idx].matrix # 需要转为array
        # if idx % 30 != 0 : #debug 间隔 2 10 15 30 60 100 
        #     continue
        # print('read frame: {}'.format(depth_img_path))
        
        # color_data = cv2.imread(rgb_img_path) 
        savescale = None
        if '.png' in depth_img_path:
            depth_data = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED) # replica uint16

            depth_trunc = 10.0 #655 200 100 50 25
            skydepth = depth_trunc # 65535
        elif '.npy' in depth_img_path: #对于tartanqir来说 gt depth 保存真实数值 m 0~ 10000（infinite）
            depth_data = np.load(depth_img_path) #float32
            # 实际上给的深度是有大于 10000 即使是office 所以先clip吧
            depth_data = np.clip(depth_data, 0, 10000)
            skydepth = 10000
            # depth_trunc = 25.0 # 20 25 30 50 100 1000
            # savescale = 100.
        png_depth_scale = frame_reader.calib[4] #深度图尺度
        # 保存为tmp.png 为了uint16不损失小数 
        if savescale is None: # 非tartanair
            savescale = png_depth_scale # 1. # 1 100
        if idx==0:
            print('save scale: {}'.format(savescale))
        depth_data = depth_data.astype(np.float32) / png_depth_scale #真实单位
        bf = frame_reader.calib[5] * frame_reader.calib[0]
        disp_data = bf / depth_data #depth_data _shuffle
        disp_data_wnoise = model_noise(disp_data, depth_data, bf, sigma_d=(6.5), sigma_s=1.5) # 3. 0.5 6.5
        # 整体的schuffle
        disp_data_wnoise_shuffle = disp_data_wnoise # shuffle_rand(disp_data_wnoise, sigma_s=0.75) # shuffle_rand(disp_data_wnoise, sigma_s=0.75) # 0.25 0.5 
        # 再对增加噪声的视差转为噪声深度图
        depth_data_wnoise_shuffle = bf / disp_data_wnoise_shuffle
        if (frame_reader.depth_noise and False):
            if 'ICL' in depth_img_path:
                depth_n_path = os.path.join(depth_img_path.split('/')[-5], depth_img_path.split('/')[-4],'living_room_traj0n_frei_png',depth_img_path.split('/')[-2],depth_img_path.split('/')[-1])
                depth_n = cv2.imread(depth_n_path, cv2.IMREAD_UNCHANGED)
                print('read noise from ICL data: \t', depth_n_path)
                depth_n = depth_n.astype(np.float32) / png_depth_scale #真实单位
            # 可视化 噪声图 和原深度图的对比
            depth_error = np.abs(depth_data - depth_noise)
            depth_error_n = np.abs(depth_data - depth_n)
            # 画在一张图里
            max_depth = np.max(depth_data)
            max_noise = np.max(depth_noise)
            max_dv = max(max_depth, max_noise)
            max_error = np.max(depth_error)
            plt.cla()
            
            plt.subplot(2,3,1)
            plt.title('raw depth')
            plt.imshow(depth_data)
            plt.subplot(2,3,2)
            plt.title('icl noise')
            plt.imshow(depth_n)
            plt.subplot(2,3,3)
            plt.title('icl err map')
            plt.imshow(depth_error_n)
            plt.subplot(2,3,4)
            plt.title('raw depth')
            plt.imshow(depth_data)
            plt.subplot(2,3,5)
            plt.title('noised depth')
            plt.imshow(depth_noise)
            plt.subplot(2,3,6)
            plt.title('my error map')
            plt.imshow(depth_error)
            plt.show()
    
        # skydepth = depth_trunc
        depth_data1 = copy.deepcopy(depth_data_wnoise_shuffle) # depth_noise depth_data depth_n
        # depth_data1[depth_noise > depth_trunc] = 0.0 #就0 即 不会被转为点云 float32 skydepth
        
        depth_data2 = depth_data1*savescale
        savepath = os.path.join(mydir, os.path.basename(depth_img_path))
        print('save noise at: \t', savepath)
        if '.png' in depth_img_path:
            cv2.imwrite(savepath, depth_data2.astype(np.uint16))
        elif '.npy' in depth_img_path:
            np.save(savepath, depth_data2)
        

        # # 在用o3d读取 这就是已经 clip 并转换真实单位的深度
        # depth_raw = o3d.io.read_image("tmp.png") 
        # H, W = np.asarray(depth_raw).shape
        # color_data = cv2.resize(color_data, (W, H))
        # # 保存
        # cv2.imwrite('tmp-color.jpg', color_data)
        # color_raw = o3d.io.read_image('tmp-color.jpg')
        
        # # http://www.open3d.org/docs/release/python_api/open3d.geometry.RGBDImage.html#open3d.geometry.RGBDImage.create_from_color_and_depth
        # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        #     color_raw, depth_raw, depth_scale=savescale, depth_trunc=skydepth, convert_rgb_to_intensity=False)
        # # 缩放深度1000.0 1.0 截断3.0 skydepth 到0 rgb 2 intensity True
        # if idx == 0:
        #     print(rgbd_image) 
            
        # inter = o3d.camera.PinholeCameraIntrinsic()
        # inter.set_intrinsics(W, H, frame_reader.calib[0], frame_reader.calib[1], frame_reader.calib[2], frame_reader.calib[3])    
        
        # if (True in np.isinf(gt_c2w)) or (True in np.isnan(gt_c2w)): #scannet里 会有-inf
        #     print('c2w nan or inf at ', idx)
        #     continue
        # pcd_idx = o3d.geometry.PointCloud.create_from_rgbd_image(
        # rgbd_image, inter,#) #,
        # extrinsic=np.linalg.inv(gt_c2w)) #测试这里就用外参 哦好像是得用 w2c!!
        # pcd_idx_w = pcd_idx# .transform(gt_c2w)
        
        # pcd_combined += pcd_idx_w
        # if idx >=0:
        #     break
        
    
    # #获取真实bound
    # # 计算轴对齐边界框
    # abox = pcd_combined.get_axis_aligned_bounding_box()
    # print('axis_aligned_bounding_box: \n', abox.get_print_info())
    # print('extent: \n', abox.get_extent())
    
    # pcdarray = np.asarray(pcd_combined.points)
    # print('pcd shape: \n', pcdarray.shape) #(n,3)

    # #保存最终点云
    # pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=0.05) # .voxel_down_sample(voxel_size=0.7) 0.4 0.15 0.2 0.01 0.02
    # # o3d.io.write_point_cloud(os.path.join(frame_reader.path,"firstiso-gtdt25.ply"), pcd_combined_down)
    # o3d.io.write_point_cloud("icl-nf0.ply", pcd_combined_down)
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    # size=1.0, origin=[0, 0, 0]) #显示坐标系 1.0 20.0
    # o3d.visualization.draw_geometries([pcd_combined_down, mesh_frame])
