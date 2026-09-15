"""Shared RGB-D object measurement. No entity-port/scene-truth access or motion."""
import math
import time


def localize_object_rgbd(*, entity_id, query, acquire, tf, camera_info, model,
                         bbox_query, min_points=15, after=0.0, compact=False, sync_tolerance_s=.25):
    from harness.skills.rgbd_skills import localize_person_in_camera, point_to_parent

    def failure(status, reason):
        return {'operation_ok': False, 'task_status': status, 'error': reason}

    pair = acquire(after=after)
    if pair is None:
        return failure('observation_unavailable', '缺少同步 RGB-D')
    color, depth = pair
    timestamp = float(depth.ts)
    transform = tf.get('world', depth.frame_id or 'camera_optical', timestamp,
                       time_tolerance=sync_tolerance_s, forward_tolerance=.5)
    if transform is None:
        return failure('observation_unavailable', '缺少该帧的相机世界变换')
    try:
        bbox = bbox_query(model, color, query)
    except Exception as exc:
        return failure('observation_unavailable', f'视觉请求失败：{type(exc).__name__}' if compact else f'{type(exc).__name__}: {exc}'[:500])
    if bbox is None:
        return failure('target_not_found', '当前视野未检测到指定水瓶')
    if (len(bbox) != 4 or not all(math.isfinite(float(v)) for v in bbox)
            or not 0 <= bbox[0] < bbox[2] <= depth.data.shape[1]
            or not 0 <= bbox[1] < bbox[3] <= depth.data.shape[0]):
        return failure('verification_failed', '视觉框越界或无效')
    estimate = localize_person_in_camera(bbox, depth.data, camera_info,
        min_points=min_points, cluster_radius_m=.08 if compact else .35)
    if estimate is None or (compact and estimate.depth_spread_m > .06):
        return failure('verification_failed', '水瓶框内没有可靠深度，不能用二维框推测位置')
    point = point_to_parent(transform, estimate.point)
    visual = {'entity_id': entity_id, 'source': 'robot_rgbd', 'frame_timestamp': timestamp,
              'color_timestamp': float(color.ts), 'bbox': [float(v) for v in bbox],
              'depth_m': float(estimate.depth_m), 'valid_depth_points': int(estimate.valid_points),
              'depth_spread_m': float(estimate.depth_spread_m),
              'pose': [float(point.x), float(point.y), float(point.z)], 'observed_at': time.time()}
    return {'operation_ok': True, 'task_status': 'object_localized', 'visual': visual}
