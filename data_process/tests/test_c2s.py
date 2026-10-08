"""Contact2SIV 单元测试 + 本机真实数据冒烟。

纯几何/构建测试不需要 GPU、不需要数据集：
  python -m unittest discover -s tests -v
真实数据冒烟需要本机数据集（TEST_REAL_DATA=1）：
  TEST_REAL_DATA=1 python -m unittest tests.test_c2s.RealDataSmoke -v
"""
from __future__ import annotations

import math
import os
import unittest

import numpy as np
from c2s.contacts import (
    ContactFrame,
    ContactPoint,
    EEFContactConfig,
    EEFContactEstimator,
    RobotwinSimContactSource,
    read_jsonl,
    write_jsonl,
)
from c2s.geometry import Camera
from c2s.siv import SIVBuilder, SIVConfig
from c2s.spec import (
    ARM_SLICES,
    GRIPPER_TIP_LEN,
    HEAD_CAMERA,
    ROBOTWIN_LEROBOT_ROOT,
    eef_to_tip_pose,
    quat_wxyz_to_R,
)


class TestGeometry(unittest.TestCase):
    def test_head_camera_intrinsics(self):
        cam = Camera.head_camera(image_size=(640, 480))
        f = (480 / 2.0) / math.tan(math.radians(37.0) / 2.0)
        self.assertAlmostEqual(cam.K[0, 0], f, places=4)
        self.assertAlmostEqual(cam.K[0, 2], 320.0, places=6)

    def test_camera_center_projects_to_principal_point(self):
        """相机正前方 1m 的点应投到主点；这是坐标系没写反的最基本检查。"""
        cam = Camera.head_camera(image_size=(640, 480))
        pos = HEAD_CAMERA["position"]
        fwd = HEAD_CAMERA["forward"] / np.linalg.norm(HEAD_CAMERA["forward"])
        px, depth, valid = cam.project((pos + fwd * 1.0)[None, :])
        self.assertTrue(valid[0])
        self.assertAlmostEqual(px[0, 0], 320.0, places=3)
        self.assertAlmostEqual(px[0, 1], 240.0, places=3)
        self.assertAlmostEqual(depth[0], 1.0, places=4)

    def test_behind_camera_is_invalid(self):
        cam = Camera.head_camera(image_size=(640, 480))
        pos = HEAD_CAMERA["position"]
        fwd = HEAD_CAMERA["forward"] / np.linalg.norm(HEAD_CAMERA["forward"])
        _, _, valid = cam.project((pos - fwd * 1.0)[None, :])
        self.assertFalse(valid[0])

    def test_pixel_scaling_to_map_size(self):
        cam = Camera.head_camera(image_size=(320, 256))
        pos = HEAD_CAMERA["position"]
        fwd = HEAD_CAMERA["forward"] / np.linalg.norm(HEAD_CAMERA["forward"])
        px, _, valid = cam.project((pos + fwd * 1.0)[None, :])
        self.assertTrue(valid[0])
        self.assertAlmostEqual(px[0, 0], 160.0, places=3)
        self.assertAlmostEqual(px[0, 1], 128.0, places=3)

    def test_sigma_is_depth_adaptive(self):
        cam = Camera.head_camera(image_size=(320, 256))
        s_near = cam.sigma_px(np.array([0.5]), 0.02)
        s_far = cam.sigma_px(np.array([2.0]), 0.02)
        self.assertGreater(s_near[0], s_far[0])          # 近处核更宽
        self.assertGreaterEqual(s_near[0], 2.0)
        self.assertLessEqual(s_far[0], 40.0)

    def test_quat_and_tip_pose(self):
        R = quat_wxyz_to_R(np.array([1.0, 0, 0, 0]))
        self.assertTrue(np.allclose(R, np.diag([1.0, -1.0, -1.0])))
        tip, _ = eef_to_tip_pose(np.zeros(3), np.array([1.0, 0, 0, 0]))
        self.assertAlmostEqual(float(np.linalg.norm(tip)), GRIPPER_TIP_LEN, places=6)


class TestContacts(unittest.TestCase):
    def _fake_scene(self, points):
        class P:
            def __init__(self, p):
                self.position = np.asarray(p, dtype=np.float64)

        class Body:
            def __init__(self, name):
                self.entity = type("E", (), {"name": name})()

        class C:
            def __init__(self, pts, names):
                self.bodies = [Body(n) for n in names]
                self.points = [P(p) for p in pts]

        class Scene:
            def get_contacts(self):
                return [C(points, ["fl_link7", "object_0"]),
                        C([[9.0, 9.0, 9.0]], ["table", "object_0"])]  # 无夹爪 -> 忽略

        return Scene()

    def _fake_scene_multi(self, specs):
        """specs: [(points, names), ...] -> 多接触对的假 scene。"""
        class P:
            def __init__(self, p):
                self.position = np.asarray(p, dtype=np.float64)

        class Body:
            def __init__(self, name):
                self.entity = type("E", (), {"name": name})()

        class C:
            def __init__(self, pts, names):
                self.bodies = [Body(n) for n in names]
                self.points = [P(p) for p in pts]

        class Scene:
            def get_contacts(self):
                return [C(pts, names) for pts, names in specs]

        return Scene()

    def test_sim_source_filters_non_gripper_contacts(self):
        src = RobotwinSimContactSource(["fl_link7"], max_points=8)
        scene = self._fake_scene([[0.1, 0.0, 0.9], [0.1, 0.01, 0.9]])
        pts = src.extract(scene)
        self.assertEqual(len(pts), 2)
        self.assertTrue(all(p.pair == ["fl_link7", "object_0"] for p in pts))
        self.assertTrue(all(0.2 <= p.weight <= 1.0 for p in pts))

    def test_sim_source_drops_self_and_camera_contacts(self):
        """两指互碰、夹爪↔腕相机（刚性安装，每帧都在）都必须剔除。"""
        src = RobotwinSimContactSource(["fl_link7", "fl_link8"], max_points=8)
        pts = src.extract(self._fake_scene_multi([
            ([[0.1, 0, 0.9]], ["fl_link7", "fl_link8"]),        # 两指互碰 -> 丢
            ([[0.1, 0, 0.9]], ["fl_link7", "left_camera"]),      # 腕相机 -> 丢
            ([[0.1, 0, 0.9]], ["fl_link7", "link_1"]),           # 真接触 -> 留
            ([[0.9, 0, 0.9]], ["link_0", "link_1"]),             # 物体自接触 -> 默认丢
        ]))
        self.assertEqual(len(pts), 1)
        self.assertEqual(pts[0].pair, ["fl_link7", "link_1"])
        self.assertEqual(pts[0].kind, "grasp")
        # 打开 include_object_pairs 后物体间接触也会被保留
        src2 = RobotwinSimContactSource(["fl_link7", "fl_link8"], max_points=8,
                                        include_object_pairs=True)
        pts2 = src2.extract(self._fake_scene_multi([
            ([[0.9, 0, 0.9]], ["link_0", "link_1"]),
        ]))
        self.assertEqual(len(pts2), 1)
        self.assertEqual(pts2[0].kind, "support")

    def test_sim_source_downsample(self):
        src = RobotwinSimContactSource(["fl_link7"], max_points=4)
        scene = self._fake_scene(np.random.rand(50, 3))
        self.assertEqual(len(src.extract(scene)), 4)

    def test_eef_estimator_detects_grasp_and_release(self):
        T = 20
        eef = np.zeros((T, 16), dtype=np.float64)
        eef[:, 3] = 1.0        # left quat w=1
        eef[:, 11] = 1.0       # right quat w=1
        eef[:, 0] = 0.0
        eef[:, 2] = 0.9
        eef[:, 7] = 1.0        # left gripper open
        eef[:, 15] = 1.0       # right gripper open
        eef[5:12, 7] = 0.0     # left closed (grasp)
        est = EEFContactEstimator(EEFContactConfig())
        frames = est(eef)
        self.assertEqual(len(frames), T)
        self.assertEqual(len(frames[0].contacts), 0)
        self.assertTrue(all(c.kind == "grasp" for c in frames[6].contacts))
        self.assertEqual(len(frames[6].contacts), 2)      # 双指
        # 释放帧(12) 起 release_window 内出现 place 接触
        kinds = {c.kind for c in frames[12].contacts}
        self.assertIn("place", kinds)
        self.assertEqual(len(frames[19].contacts), 0)     # 窗口结束

    def test_jsonl_roundtrip(self, ):
        f = [ContactFrame(0, [ContactPoint([0.1, 0.2, 0.3], 0.8, ["a", "b"], "grasp")])]
        p = "/tmp/_c2s_test_contacts.jsonl"
        write_jsonl(f, p)
        g = read_jsonl(p)
        self.assertEqual(g[0].frame, 0)
        self.assertAlmostEqual(g[0].contacts[0].position[1], 0.2)
        self.assertEqual(g[0].contacts[0].pair, ["a", "b"])


class TestSIV(unittest.TestCase):
    def _cam(self, size=(320, 256)):
        return Camera.head_camera(image_size=size)

    def _frames(self, n=5):
        pos = HEAD_CAMERA["position"]
        fwd = HEAD_CAMERA["forward"] / np.linalg.norm(HEAD_CAMERA["forward"])
        base = pos + fwd * 1.0
        out = []
        for t in range(n):
            out.append(ContactFrame(t, [ContactPoint((base + np.array([0.01 * t, 0.0, 0.0])).tolist(),
                                                     1.0, ["g", "o"], "grasp")]))
        return out

    def test_map_shape_and_range(self):
        res = SIVBuilder(self._cam(), SIVConfig()).build(self._frames())
        self.assertEqual(res.maps.shape, (5, 256, 320))
        self.assertGreaterEqual(res.maps.min(), 0.0)
        self.assertLessEqual(res.maps.max(), 1.0)
        self.assertAlmostEqual(float(res.maps.max()), 1.0, places=5)  # 归一化后峰值为 1

    def test_peak_near_projected_contact(self):
        cam = self._cam()
        frames = self._frames(1)
        res = SIVBuilder(cam, SIVConfig()).build(frames)
        v, u = np.unravel_index(np.argmax(res.maps[0]), res.maps[0].shape)
        px, _, valid = cam.project(np.asarray(frames[0].contacts[0].position)[None, :])
        self.assertTrue(valid[0])
        self.assertLess(abs(u - px[0, 0]), 3.0)
        self.assertLess(abs(v - px[0, 1]), 3.0)

    def test_depth_adaptive_vs_fixed_sigma(self):
        cam = self._cam()
        pos = HEAD_CAMERA["position"]
        fwd = HEAD_CAMERA["forward"] / np.linalg.norm(HEAD_CAMERA["forward"])
        near = ContactFrame(0, [ContactPoint((pos + fwd * 0.5).tolist(), 1.0)])
        far = ContactFrame(1, [ContactPoint((pos + fwd * 2.0).tolist(), 1.0)])
        fixed = SIVBuilder(cam, SIVConfig(fixed_sigma=10.0)).build([near, far])
        adapt = SIVBuilder(cam, SIVConfig()).build([near, far])
        area = lambda m: int((m > 0.5).sum())
        self.assertGreater(area(adapt.maps[0]), area(adapt.maps[1]))   # 近 > 远
        self.assertEqual(area(fixed.maps[0]), area(fixed.maps[1]))     # 固定 sigma 面积相同

    def test_depth_consistency_filter(self):
        cam = self._cam()
        fwd = HEAD_CAMERA["forward"] / np.linalg.norm(HEAD_CAMERA["forward"])
        p = HEAD_CAMERA["position"] + fwd * 0.6
        dm = np.full((256, 320), 3.0, dtype=np.float32)             # 深度图说这里 3m
        f = ContactFrame(0, [ContactPoint(p.tolist(), 1.0)], depth_path="x")
        res = SIVBuilder(cam, SIVConfig(depth_tolerance=0.05)).build(
            [f], depth_loader=lambda _p: dm)
        self.assertEqual(res.stats["projected"], 0)
        self.assertEqual(res.stats["depth_rejected"], 1)

    def test_empty_contacts_give_zero_map(self):
        res = SIVBuilder(self._cam(), SIVConfig()).build([ContactFrame(0, [])])
        self.assertEqual(float(res.maps[0].max()), 0.0)
        self.assertEqual(res.stats["empty_frames"], 1)


@unittest.skipUnless(os.environ.get("TEST_REAL_DATA") == "1",
                     "需要本机 RoboTwin 数据集")
class RealDataSmoke(unittest.TestCase):
    """跑真实 episode：投影 sanity check + SIV 生成。"""

    def test_real_episode(self):
        from c2s.contacts import EEFContactEstimator
        from c2s.lerobot_io import read_episode, read_frames
        from c2s.siv import SIVBuilder, SIVConfig
        from c2s.spec import CAM_KEYS

        task_dir = os.path.join(ROBOTWIN_LEROBOT_ROOT,
                                "lift_pot-aloha-agilex_randomized_500-1000")
        if not os.path.isdir(task_dir):
            self.skipTest(f"数据集不存在: {task_dir}")
        eef = read_episode(task_dir, 0)
        self.assertEqual(eef.shape[1], 16)
        cam = Camera.head_camera(image_size=(320, 256))
        frames = EEFContactEstimator()(eef)
        res = SIVBuilder(cam, SIVConfig()).build(frames)
        self.assertEqual(res.maps.shape[0], len(eef))
        self.assertGreater(res.stats["projected"], 0)
        # 有接触的帧，SIV 峰值必须和投影点一致
        t = next(i for i, f in enumerate(frames) if f.contacts)
        px, _, valid = cam.project(np.asarray([c.position for c in frames[t].contacts]))
        px = px[valid]
        self.assertTrue(len(px) > 0)
        v, u = np.unravel_index(np.argmax(res.maps[t]), res.maps[t].shape)
        self.assertLess(min(np.linalg.norm(np.array([u, v]) - p) for p in px), 5.0)
        # 投影 sanity check：抓取阶段（gripper 闭合、臂在桌面上方）夹爪尖应落在图像内。
        # home 位姿在 head camera 视野外属正常，不能拿第 0 帧判断。
        from c2s.geometry import sanity_check_eef
        cam_full = Camera.head_camera(image_size=(640, 480))
        t = int(np.argmin(eef[:, 7]))
        out = sanity_check_eef(cam_full, eef[t])
        for arm, p in out.items():
            self.assertIsNotNone(p, f"{arm} 未投到图像内 (frame {t})")
            self.assertTrue(0 <= p[0] < 640 and 0 <= p[1] < 480)


if __name__ == "__main__":
    unittest.main(verbosity=2)
