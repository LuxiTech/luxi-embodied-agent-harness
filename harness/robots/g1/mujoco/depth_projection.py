"""Align the pinned depth projector with MuJoCo's pixel-centre convention."""
import math
import numpy as np


def pixel_center_projection(original):
    def project(depth_image, camera_pos, camera_mat, fov_degrees=120):
        points = original(depth_image, camera_pos, camera_mat, fov_degrees=fov_degrees)
        if not len(points):
            return points
        rotation = np.asarray(camera_mat).reshape(3, 3)
        # The upstream projector samples integer pixels at width/2,height/2.
        # MuJoCo renders pixel centres: u+0.5,v+0.5. Preserve upstream filtering.
        camera = (np.asarray(points) - np.asarray(camera_pos)) @ rotation
        depth = -camera[:, 2]
        focal = depth_image.shape[0] / (2 * math.tan(math.radians(fov_degrees) / 2))
        offset = depth / (2 * focal)
        correction = np.column_stack((offset, -offset, np.zeros_like(offset)))
        return points + correction @ rotation.T
    return project
