#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos


class Scene:
    def __init__(self, model_path, args, gaussians, load_iteration=None, resolution_scales=[1.0], create_gs_now=True):
        self.model_path = model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        # if not self.loaded_iter:
        #     with open(scene_info.ply_path, "rb") as src_file, open(os.path.join(self.model_path, "input.ply"), "wb") as dest_file:
        #         dest_file.write(src_file.read())

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        self.num_train_cameras = len(scene_info.train_cameras)
        self.num_test_cameras = len(scene_info.test_cameras)

        self.gaussians.set_embed(self.num_train_cameras)
        if create_gs_now:
            if self.loaded_iter:
                self.load(self.loaded_iter)
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def load(self, iteration=None, ckpt_path=None):
            if iteration:
                self.loaded_iter = iteration
                self.gaussians.load_ply(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply"))
                self.gaussians.load_mlp_checkpoints(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter)))
            elif ckpt_path:
                self.gaussians.load_ply(os.path.join(ckpt_path, "point_cloud.ply"))
                self.gaussians.load_mlp_checkpoints(ckpt_path)
            else:
                assert False, "No iteration or checkpoint path provided!"

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, f"point_cloud/iteration_{iteration}")
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_mlp_checkpoints(point_cloud_path)
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
