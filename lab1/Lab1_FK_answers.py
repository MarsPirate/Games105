import numpy as np
from scipy.spatial.transform import Rotation as R


def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1, -1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    from bvh import BvhNode, Bvh

    with open(bvh_file_path) as f:
        mocap = Bvh(f.read())

    joint_name = [joint.name for joint in mocap.get_joints()]
    joint_parent = [mocap.joint_parent_index(joint.name) for joint in mocap.get_joints()]
    joint_offset = np.array([mocap.joint_offset(joint.name) for joint in mocap.get_joints()])
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    from scipy.spatial.transform import Rotation as R
    frame_data = motion_data[frame_id]

    def _cal_global_pos_rot():
        root_pos = frame_data[0:3]
        root_rot = frame_data[3:6]
        g_pos = []
        g_rot = []

        # prepare local rotation
        local_rot = []
        for i in range(len(joint_parent)):
            rot = frame_data[3+i*3:3+i*3+3]
            local_rot.append(rot)

        # from local to global
        for i in range(len(joint_offset)):
            if 0 == i:
                g_pos.append(root_pos)
                g_rot.append(root_rot)
                continue
            p_m = R.from_euler("XYZ", g_rot[joint_parent[i]], degrees=True).as_matrix()
            l_m = R.from_euler("XYZ", local_rot[i], degrees=True).as_matrix()
            rot = R.from_matrix(np.dot(p_m, l_m)).as_euler("XYZ", degrees=True)
            pos = g_pos[joint_parent[i]] + np.dot(p_m, joint_offset[i])
            g_pos.append(pos)
            g_rot.append(rot)

        tmp_rot = []
        for rot in g_rot:
            tmp_rot.append(R.from_euler("XYZ", rot, degrees=True).as_quat())
        g_rot = tmp_rot

        return np.array(g_pos), np.array(g_rot)

    joint_positions, joint_orientations = _cal_global_pos_rot()
    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    motion_data = None
    return motion_data
