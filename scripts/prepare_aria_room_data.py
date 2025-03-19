import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from projectaria_tools.core import calibration
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.adt import (AriaDigitalTwinDataPathsProvider,
                                            AriaDigitalTwinDataProvider)
from tqdm import tqdm

DATASET_DICT = {
    "room0": {
        "agent_0": {
            "scene": "Apartment_release_decoration_seq136_M1292",
            "start": 0,
            "end": 500
        },
        "agent_1": {
            "scene": "Apartment_release_decoration_seq137_M1292",
            "start": 2070,
            "end": 2570
        },
        "agent_2": {
            "scene": "Apartment_release_decoration_seq139_M1292",
            "start": 2200,
            "end": 2700
        }
    },
    "room1": {
        "agent_0": {
            "scene": "Apartment_release_decoration_seq134_M1292",
            "start": 1500,
            "end": 2000
        },
        "agent_1": {
            "scene": "Apartment_release_decoration_skeleton_seq133_M1292",
            "start": 2230,
            "end": 2730
        },
        "agent_2": {
            "scene": "Apartment_release_decoration_skeleton_seq135_M1292",
            "start": 2200,
            "end": 2700
        }
    }
}


TEST_DATASET_DICT = {
    "room0": {
        "scene": "Apartment_release_decoration_seq133_M1292",
        "start": 2100,
        "end": 2300
    },
    "room1": {
        "scene": "Apartment_release_decoration_seq139_M1292",
        "start": 115,
        "end": 315
    }
}


def get_args():
    parser = argparse.ArgumentParser(
        description='Arguments to compute the mesh')
    parser.add_argument('agent_data_path', type=str, help='Path to the aria sequences')
    parser.add_argument('output_path', type=str, help='Output path to the resulting data')
    return parser.parse_args()


def get_valid_timestamps(data_provider, stream_id):
    valid_timestamps = []
    start, end = data_provider.get_start_time_ns(), data_provider.get_end_time_ns()
    for timestamp in data_provider.get_aria_device_capture_timestamps_ns(stream_id):
        if timestamp >= start and timestamp <= end:
            valid_timestamps.append(timestamp)
    return valid_timestamps


def get_dynamic_instances(data_provider):
    instance_ids = data_provider.get_instance_ids()
    dynamic_instance_ids = []
    for instance_id in instance_ids:
        instance_info = data_provider.get_instance_info_by_id(instance_id)
        if instance_info.motion_type == instance_info.motion_type.DYNAMIC:
            dynamic_instance_ids.append(instance_id)
    return dynamic_instance_ids


def prepare_data(data_path, output_path):

    output_path.mkdir(parents=True, exist_ok=True)

    for room_name in DATASET_DICT.keys():
        for agent_id in DATASET_DICT[room_name].keys():

            scene_path = data_path / DATASET_DICT[room_name][agent_id]["scene"]
            start_frame_id = DATASET_DICT[room_name][agent_id]["start"]
            end_frame_id = DATASET_DICT[room_name][agent_id]["end"]

            agent_output_path = output_path / room_name / f"{agent_id}"
            agent_output_path.mkdir(parents=True, exist_ok=True)
            (agent_output_path / "results").mkdir(parents=True, exist_ok=True)

            paths_provider = AriaDigitalTwinDataPathsProvider(str(scene_path))
            data_paths = paths_provider.get_datapaths_by_device_num(0)
            gt_provider = AriaDigitalTwinDataProvider(data_paths)
            stream_id = StreamId("214-1")

            sensor_name = gt_provider.raw_data_provider_ptr().get_label_from_stream_id(stream_id)
            device_calib = gt_provider.raw_data_provider_ptr().get_device_calibration()
            src_calib = device_calib.get_camera_calib(sensor_name)
            dst_calib = calibration.get_linear_camera_calibration(
                512, 512, 280, sensor_name, src_calib.get_transform_device_camera())
            rotated_calib = calibration.rotate_camera_calib_cw90deg(dst_calib)
            T_Device_Cam = rotated_calib.get_transform_device_camera()

            # dynamic_instance_ids = get_dynamic_instances(gt_provider)
            valid_timestamps = get_valid_timestamps(gt_provider, stream_id)
            c2ws = []
            for frame_id, timestamp in tqdm(enumerate(valid_timestamps[start_frame_id:end_frame_id])):
                # print(frame_id, timestamp)
                # color_data = gt_provider.get_synthetic_image_by_timestamp_ns(timestamp, stream_id)
                color_data = gt_provider.get_aria_image_by_timestamp_ns(timestamp, stream_id)
                assert color_data.is_valid(), f"Invalid color data for index {frame_id}"
                color_data = color_data.data().to_numpy_array()

                depth_data = gt_provider.get_depth_image_by_timestamp_ns(timestamp, stream_id)
                assert depth_data.is_valid(), f"Invalid depth data for index {frame_id}"
                depth_data = depth_data.data().to_numpy_array()

                # Rectify the data
                rectified_color_data = calibration.distort_by_calibration(color_data, dst_calib, src_calib)
                rectified_depth_data = calibration.distort_by_calibration(depth_data, dst_calib, src_calib)

                # Rotate the data
                rotated_color_data = np.rot90(rectified_color_data, 3)

                rotated_depth_data = np.rot90(rectified_depth_data, 3)
                rotated_depth_data = rotated_depth_data / 1000.0  # mm to m
                image_depth_data = rotated_depth_data / 10.0  # scale between 0 and 1 with the max depth value
                image_depth_data = (image_depth_data * 65535).astype(np.uint16)  # prepare for storage as png

                Image.fromarray(rotated_color_data).save(agent_output_path / "results" / f"frame{frame_id:06}.jpg")
                Image.fromarray(image_depth_data).save(agent_output_path / "results" / f"depth{frame_id:06}.png")
                # np.save(agent_output_path / "results" / f"{frame_id:06}.npy", rotated_depth_data)

                pose = gt_provider.get_aria_3d_pose_by_timestamp_ns(timestamp)
                assert pose.is_valid(), f"Invalid pose data for index {frame_id}"
                T_Scene_Device = pose.data().transform_scene_device
                c2w = T_Scene_Device @ T_Device_Cam
                c2w = c2w.to_matrix()
                # print(c2w, "\n")
                c2ws.append(c2w)

            pose_output_file = open(str(agent_output_path / "traj.txt"), "w")
            for c2w in c2ws:
                line = " ".join([str(num) for num in c2w.flatten()])
                pose_output_file.write(line + "\n")
            pose_output_file.close()


def prepare_test_data(data_path, output_path):

    output_path.mkdir(parents=True, exist_ok=True)

    for room_name in TEST_DATASET_DICT.keys():
        scene_path = data_path / TEST_DATASET_DICT[room_name]["scene"]
        start_frame_id = TEST_DATASET_DICT[room_name]["start"]
        end_frame_id = TEST_DATASET_DICT[room_name]["end"]

        agent_output_path = output_path / room_name
        agent_output_path.mkdir(parents=True, exist_ok=True)
        (agent_output_path / "results").mkdir(parents=True, exist_ok=True)

        paths_provider = AriaDigitalTwinDataPathsProvider(str(scene_path))
        data_paths = paths_provider.get_datapaths_by_device_num(0)
        gt_provider = AriaDigitalTwinDataProvider(data_paths)
        stream_id = StreamId("214-1")

        sensor_name = gt_provider.raw_data_provider_ptr().get_label_from_stream_id(stream_id)
        device_calib = gt_provider.raw_data_provider_ptr().get_device_calibration()
        src_calib = device_calib.get_camera_calib(sensor_name)
        dst_calib = calibration.get_linear_camera_calibration(
            512, 512, 280, sensor_name, src_calib.get_transform_device_camera())
        rotated_calib = calibration.rotate_camera_calib_cw90deg(dst_calib)
        T_Device_Cam = rotated_calib.get_transform_device_camera()

        # dynamic_instance_ids = get_dynamic_instances(gt_provider)
        valid_timestamps = get_valid_timestamps(gt_provider, stream_id)
        c2ws = []
        for frame_id, timestamp in tqdm(enumerate(valid_timestamps[start_frame_id:end_frame_id])):
            # print(frame_id, timestamp)
            # color_data = gt_provider.get_synthetic_image_by_timestamp_ns(timestamp, stream_id)
            color_data = gt_provider.get_aria_image_by_timestamp_ns(timestamp, stream_id)
            assert color_data.is_valid(), f"Invalid color data for index {frame_id}"
            color_data = color_data.data().to_numpy_array()

            depth_data = gt_provider.get_depth_image_by_timestamp_ns(timestamp, stream_id)
            assert depth_data.is_valid(), f"Invalid depth data for index {frame_id}"
            depth_data = depth_data.data().to_numpy_array()

            # Rectify the data
            rectified_color_data = calibration.distort_by_calibration(color_data, dst_calib, src_calib)
            rectified_depth_data = calibration.distort_by_calibration(depth_data, dst_calib, src_calib)

            # Rotate the data
            rotated_color_data = np.rot90(rectified_color_data, 3)

            rotated_depth_data = np.rot90(rectified_depth_data, 3)
            rotated_depth_data = rotated_depth_data / 1000.0  # mm to m
            image_depth_data = rotated_depth_data / 10.0  # scale between 0 and 1 with the max depth value
            image_depth_data = (image_depth_data * 65535).astype(np.uint16)  # prepare for storage as png

            Image.fromarray(rotated_color_data).save(agent_output_path / "results" / f"frame{frame_id:06}.jpg")
            Image.fromarray(image_depth_data).save(agent_output_path / "results" / f"depth{frame_id:06}.png")
            # np.save(agent_output_path / "results" / f"{frame_id:06}.npy", rotated_depth_data)

            pose = gt_provider.get_aria_3d_pose_by_timestamp_ns(timestamp)
            assert pose.is_valid(), f"Invalid pose data for index {frame_id}"
            T_Scene_Device = pose.data().transform_scene_device
            c2w = T_Scene_Device @ T_Device_Cam
            c2w = c2w.to_matrix()
            # print(c2w, "\n")
            c2ws.append(c2w)

        pose_output_file = open(str(agent_output_path / "traj.txt"), "w")
        for c2w in c2ws:
            line = " ".join([str(num) for num in c2w.flatten()])
            pose_output_file.write(line + "\n")
        pose_output_file.close()


if __name__ == "__main__":
    args = get_args()
    prepare_data(Path(args.agent_data_path), Path(args.output_path))
    prepare_test_data(Path(args.agent_data_path), Path(args.output_path) / "test")
