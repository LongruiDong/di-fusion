import importlib
import open3d as o3d
import argparse, os
import logging
import time
import torch
from utils import exp_util, vis_util
# -*- coding:utf-8 -*-
from network import utility
import numpy as np
from system import map
import system.tracker
o3d.visualization.webrtc_server.enable_webrtc()
vis_param = argparse.Namespace()
vis_param.n_left_steps = 0
vis_param.args = None
vis_param.mesh_updated = True # ？
vis_param.seq_package = None # 全局变量
vis_param.last_mesh = None


def key_step(vis):
    vis_param.n_left_steps = 1
    return False


def key_continue(vis):
    vis_param.n_left_steps += 10000 # 控制运行最大的帧数？
    return False


def update_geometry(geom, name, vis):
    if not isinstance(geom, list):
        geom = [geom]

    if name in vis_param.__dict__.keys():
        for t in vis_param.__dict__[name]:
            vis.remove_geometry(t, reset_bounding_box=False)
    for t in geom:
        vis.add_geometry(t, reset_bounding_box=False)
    vis_param.__dict__[name] = geom


def refresh(vis):
    if vis:
        # This spares slots for meshing thread to emit commands.
        time.sleep(0.02)

    if not vis_param.mesh_updated and vis_param.args.run_async:
        map_mesh = vis_param.map.extract_mesh(vis_param.args.resolution, 0, extract_async=True)
        if map_mesh is not None:
            vis_param.mesh_updated = True
            update_geometry(map_mesh, "mesh_geometry", vis)

    if vis_param.n_left_steps == 0:
        return False
    if vis_param.sequence.frame_id >= len(vis_param.sequence):
        if vis:
            return False
        else:
            raise StopIteration

    vis_param.n_left_steps -= 1

    logging.info(f"Frame ID = {vis_param.sequence.frame_id}")
    frame_data = next(vis_param.sequence) # 读入一一帧数据

    # Prune invalid depths 在给定区间外的就为无效值
    frame_data.depth[torch.logical_or(frame_data.depth < vis_param.args.depth_cut_min,
                                      frame_data.depth > vis_param.args.depth_cut_max)] = np.nan

    # Do tracking. 通过第四个参数固定gt pose? vis_param.sequence.gt_trajectory[vis_param.sequence.frame_id-1]
    # frame_pose = vis_param.tracker.track_camera(frame_data.rgb, frame_data.depth, frame_data.calib, # 原始
    #                                             vis_param.sequence.first_iso if len(vis_param.tracker.all_pd_pose) == 0 else None)
    gt_c2w = vis_param.sequence.gt_trajectory[vis_param.sequence.frame_id-1].matrix # None # vis_param.sequence.gt_trajectory[vis_param.sequence.frame_id-1].matrix
    if (True in np.isinf(gt_c2w)) or (True in np.isnan(gt_c2w)): # (True in np.isinf(gt_c2w)) or (True in np.isnan(gt_c2w)): #scannet里 会有-inf False
        print('gt_c2w nan or inf at frame-', vis_param.sequence.frame_id-1)
        frame_pose = vis_param.tracker.track_camera(frame_data.rgb, frame_data.depth, frame_data.calib,
                                                    None) # 那就自己优化来track
    else:
        # if vis_param.sequence.frame_id-1 == 0:
        #     print('tracking on..')
        frame_pose = vis_param.tracker.track_camera(frame_data.rgb, frame_data.depth, frame_data.calib, # None)
                                                    # vis_param.sequence.first_iso if len(vis_param.tracker.all_pd_pose) == 0 else None)
                                                    # vis_param.sequence.gt_trajectory[vis_param.sequence.frame_id-1] if len(vis_param.tracker.all_pd_pose) == 0 else None) # 本身gt pose 首帧就是first_iso 等效的
                                                    vis_param.sequence.gt_trajectory[vis_param.sequence.frame_id-1] ) # 对于tartanair firstiso 不是gt[0] 因为没有归一化 但通过gt 来固定pose
                                                    # vis_param.sequence.gt_trajectory[vis_param.sequence.frame_id-1] if vis_param.sequence.frame_id-1 == 0 else None) # for tracking
    tracker_pc, tracker_normal = vis_param.tracker.last_processed_pc

    if vis:
        pc_geometry = vis_util.pointcloud(frame_pose @ tracker_pc.cpu().numpy())
        update_geometry(pc_geometry, "pc_geometry", vis)
        update_geometry(vis_util.frame(), "frame", vis)
        update_geometry(vis_util.trajectory([t.t for t in vis_param.tracker.all_pd_pose]), "traj_geometry", vis)
        update_geometry(vis_util.camera(frame_pose, scale=0.15, color_id=3), "camera_geometry", vis)

    if (vis_param.sequence.frame_id - 1) % vis_param.args.integrate_interval == 0: # 默认每integrate_interval=20 做一次mapping
        opt_depth = frame_pose @ tracker_pc
        opt_normal = frame_pose.rotation @ tracker_normal
        vis_param.map.integrate_keyframe(opt_depth, opt_normal, async_optimize=vis_param.args.run_async,
                                         do_optimize=False)
        map_mesh = vis_param.map.extract_mesh(vis_param.args.resolution, int(4e6), max_std=0.5, # 4e8 max_std 0.15(ICL) 2000.0 会影响mesh抽取的 过滤阈值？0.45 10.45
                                                  extract_async=vis_param.args.run_async, interpolate=True) # 放这里便于调试 即使不vis
        if vis:
            fast_preview_vis = vis_param.map.get_fast_preview_visuals()
            update_geometry(fast_preview_vis[0], "block_geometry", vis)
            update_geometry((vis_util.wireframe_bbox(vis_param.map.bound_min.cpu().numpy(), #可视化时 最外面的灰色box既是设置的bound
                                                     vis_param.map.bound_max.cpu().numpy(), color_id=4)), "bound_box", vis)
            # map_mesh = vis_param.map.extract_mesh(vis_param.args.resolution, int(4e6), max_std=100.45, # max_std 0.15 会影响mesh抽取的 过滤阈值？0.45 10.45
            #                                       extract_async=vis_param.args.run_async, interpolate=True)
            vis_param.mesh_updated = map_mesh is not None
            if map_mesh is not None:
                # Note: This may be slow: 但打开下面两行没看到啥变化啊。。
                # map_mesh.merge_close_vertices(0.01)
                # map_mesh.compute_vertex_normals()
                update_geometry(map_mesh, "mesh_geometry", vis)
                
        if map_mesh is not None:
            vis_param.last_mesh = map_mesh #保存
        # 试着保存
        ext = '.ply' #track
        if vis_param.sequence.depth_noise:
            ext = 'noise.ply' #track-
        if vis_param.sequence.frame_id + vis_param.args.integrate_interval >= len(vis_param.sequence):
            if 'icl_nuim' in vis_param.seq_package:
                seqname = str(vis_param.sequence.path).split('/')[-1]
                savepath = 'icl-'+seqname+ext
            elif 'scannet' in vis_param.seq_package:
                seqname = str(vis_param.sequence.path).split('/')[-2]
                savepath = 'scannet-'+seqname+ext
            elif 'tartanair' in vis_param.seq_package:
                seqname = str(vis_param.sequence.path).split('/')
                savepath = 'tartanair-'+seqname[-3]+'-'+seqname[-1]+ext
            elif 'replica' in vis_param.seq_package:
                seqname = str(vis_param.sequence.path).split('/')[-1]
                savepath = 'replica-'+seqname+ext
            else:
                savepath = 'save'+ext
            print('save mesh to \t', savepath)
            if map_mesh is not None:
                o3d.io.write_triangle_mesh(savepath, map_mesh)
            else: #可能会出现空
                o3d.io.write_triangle_mesh(savepath, vis_param.last_mesh)
                
                

    return True


if __name__ == '__main__':
    parser = exp_util.ArgumentParserX()
    args = parser.parse_args() #载入外部参数
    logging.basicConfig(level=logging.INFO)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    # Load in network.  (args.model is the network specification)
    model, args_model = utility.load_model(args.training_hypers, args.using_epoch)
    args.model = args_model
    args.mapping = exp_util.dict_to_args(args.mapping) #两个模块的参数设置
    args.tracking = exp_util.dict_to_args(args.tracking)

    # Load in sequence. 数据接口
    seq_package, seq_class = args.sequence_type.split(".") #icl_nuim(data 的.py) ICLNUIMSequence 对应的接口类
    sequence_module = importlib.import_module("dataset.production." + seq_package) #dataset/production/icl_nuim.py
    sequence_module = getattr(sequence_module, seq_class) #class ICLNUIMSequence
    vis_param.sequence = sequence_module(**args.sequence_kwargs) #输入数据api class 运行init 默认不读入gt pose
    vis_param.seq_package = seq_package
    
    # Mapping
    if torch.cuda.device_count() > 1: #可以多卡？ run_async有关
        main_device, aux_device = torch.device("cuda", index=0), torch.device("cuda", index=1)
        print('Note: use 2 GPUs...')
    elif torch.cuda.device_count() == 1:
        main_device, aux_device = torch.device("cuda", index=0), None
    else:
        assert False, "You must have one GPU."
    
    vis_param.map = map.DenseIndexedMap(model, args.mapping, args.model.code_length, main_device,
                                        args.run_async, aux_device)
    vis_param.tracker = system.tracker.SDFTracker(vis_param.map, args.tracking) # 初始化track模块
    vis_param.args = args
    print('clip depth: [{},{}]'.format(vis_param.args.depth_cut_min, vis_param.args.depth_cut_max))
    if args.vis:
        # Run the engine. Internal clock driven by Open3D visualizer.
        engine = o3d.visualization.VisualizerWithKeyCallback()
        engine.create_window(window_name="Implicit SLAM", width=1280, height=720, visible=True)
        engine.register_key_callback(key=ord(","), callback_func=key_step)
        engine.register_key_callback(key=ord("."), callback_func=key_continue)
        engine.get_render_option().mesh_show_back_face = True
        engine.register_animation_callback(callback_func=refresh)
        vis_ph = vis_util.wireframe_bbox([-4., -4., -4.], [4., 4., 4.])
        engine.add_geometry(vis_ph) #vis_ph
        engine.remove_geometry(vis_ph, reset_bounding_box=False)
        engine.run()
        engine.destroy_window()
    else:
        key_continue(None)
        try:
            while True:
                refresh(None)
        except StopIteration:
            pass
