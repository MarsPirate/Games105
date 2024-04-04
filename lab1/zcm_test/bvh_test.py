# coding: utf-8
from bvh import Bvh, BvhNode


def readBvh(bvh_loc):
    with open(bvh_loc) as f:
        mocap = Bvh(f.read())

    for joint in mocap.get_joints():
        print(joint, mocap.joint_parent(joint.name), mocap.joint_parent_index(joint.name))


if __name__ == "__main__":
    bvh_loc = R"C:\Users\admin\Documents\zengchunming\Games105\GAMES-105\lab1\data\walk60.bvh"
    readBvh(bvh_loc)
