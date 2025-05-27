import cv2
import numpy as np
import pyrealsense2 as rs
class DepthCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        spatial = rs.spatial_filter()
        temporal_filter = rs.temporal_filter()
        hole_filling_filter = rs.hole_filling_filter(1)
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal_filter.process(depth_frame)
        depth_frame = hole_filling_filter.process(depth_frame)
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        if not depth_frame or not color_frame:
            return False, None, None, None
        return True, depth_image, color_image, intrinsics
    def release(self):
        self.pipeline.stop()