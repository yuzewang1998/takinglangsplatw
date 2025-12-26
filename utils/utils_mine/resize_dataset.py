import os
from PIL import Image
import numpy as np
import collections
import struct
import shutil
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """pack and write to a binary file.
    :param fid:
    :param data: data to send, if multiple elements are sent at the same time,
    they should be encapsuled either in a list or a tuple
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    should be the same length as the data list or tuple
    :param endian_character: Any of {@, =, <, >, !}
    """
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])





def read_cameras_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid,
                num_bytes=8 * num_params,
                format_char_sequence="d" * num_params,
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras

def write_cameras_binary(cameras, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for _, cam in cameras.items():
            model_id = CAMERA_MODEL_NAMES[cam.model].model_id
            camera_properties = [cam.id, model_id, cam.width, cam.height]
            write_next_bytes(fid, camera_properties, "iiQQ")
            for p in cam.params:
                write_next_bytes(fid, float(p), "d")
    return cameras


# # read cam
# with open(cameras_intrinsic_file, "rb") as fid:
#     num_cameras = read_next_bytes(fid, 8, "Q")[0]
#     for _ in range(num_cameras):
#         camera_properties = read_next_bytes(
#             fid, num_bytes=24, format_char_sequence="iiQQ")
#         camera_id = camera_properties[0]
#         model_id = camera_properties[1]
#         model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
#         width = camera_properties[2]
#         height = camera_properties[3]
#         num_params = CAMERA_MODEL_IDS[model_id].num_params
#         params = read_next_bytes(fid, num_bytes=8 * num_params,
#                                  format_char_sequence="d" * num_params)

def resize_images(input_image_folder,output_image_folder,size):
    assert os.path.exists(output_image_folder), 'Check the output folder, dont exist the output folder!'
    for filename in os.listdir(input_image_folder):
        file_path = os.path.join(input_image_folder, filename)
        # 确保只处理图片文件
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                # 打开图片
                with Image.open(file_path) as img:
                    # 调整图片大小
                    img_resized = img.resize(size)

                    # 保存到输出文件夹，文件名保持不变
                    output_path = os.path.join(output_image_folder, filename)
                    img_resized.save(output_path)
                    print(f"已处理: {filename}")
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")
                exit(-1)

# 相应的更改内参
def modify_cameras_para(cameras,new_size):
    modified_cam = {}

    for idx in cameras:
        cam = cameras[idx]
        w_scale = new_size[0]/cam[2]
        h_scale = new_size[1]/cam[3]
        m_width =  new_size[0] #'width' #1033
        h_width = new_size[1] #'height' 685
        paras = cam[4]
        paras[0] = paras[0] * w_scale
        paras[2] = paras[2] * w_scale
        paras[1] = paras[1] * h_scale
        paras[3] = paras[3] * h_scale
        modified_cam[idx] = Camera(id=cam[0], model=cam[1],
                                    width=m_width, height=h_width,
                                    params=paras)
    return modified_cam
def makedir_colmap_style(workspace_folder):
    os.makedirs(workspace_folder,exist_ok=True)
    workspace_folder = os.path.join(workspace_folder, 'dense')
    os.makedirs(workspace_folder, exist_ok=True)
    sub_folder_key = ["images","sparse","stereo"]
    for k in sub_folder_key:
        folder_path = os.path.join(workspace_folder,k)
        os.makedirs(folder_path,exist_ok=True)
if __name__ == '__main__':
    scene_name_list = ['buckingham_palace','notre_dame_front_facade','pantheon_exterior','taj_mahal','temple_nara_japan','trevi_fountain']
    # 输入文件夹和输出文件夹路径
    for scene_name in scene_name_list:
        input_workspace_folder = '/media/wangyz/DATA/UBUNTU_data/dataset/PT/'+scene_name
        output_workspace_folder = '/media/wangyz/DATA/UBUNTU_data/dataset/PT-samesize/'+scene_name
        # output_workspace_folder = '/home/wangyz/Downloads/brandenburg_gate_tiny'
        input_image_folder = os.path.join(input_workspace_folder, 'dense', 'images')  # 替换为你的输入文件夹路径
        output_image_folder = os.path.join(output_workspace_folder, 'dense', 'images')  # 替换为你的输出文件夹路径
        new_size = (512, 512)
        # cameras_extrinsic_file = os.path.join(workspace_folder, "sparse/0", "images.bin")
        input_cameras_intrinsic_file = os.path.join(input_workspace_folder, 'dense', "sparse", "cameras.bin")
        output_cameras_intrinsic_file = os.path.join(output_workspace_folder, 'dense', "sparse", "cameras.bin")
        #复制并创建文件夹格式
        makedir_colmap_style(output_workspace_folder)
        # 复制图像
        resize_images(input_image_folder,output_image_folder,new_size)
        cameras = read_cameras_binary(input_cameras_intrinsic_file)
        print(cameras)
        cameras = modify_cameras_para(cameras,new_size)
        write_cameras_binary(cameras,output_cameras_intrinsic_file)
        # copy the pcd and ext file
        shutil.copy(os.path.join(input_workspace_folder, 'dense', 'sparse','images.bin'), \
                    os.path.join(output_workspace_folder, 'dense', 'sparse','images.bin'))
        shutil.copy(os.path.join(input_workspace_folder, 'dense', 'sparse', 'points3D.bin'),\
                    os.path.join(output_workspace_folder, 'dense', 'sparse', 'points3D.bin'))