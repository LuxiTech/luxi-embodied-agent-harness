"""Worker-local setup for manipulation with the G1 model's native hands."""

from __future__ import annotations

from typing import Any
import xml.etree.ElementTree as ET


def configure_g1_native_hand(robot_xml: bytes) -> bytes:
    """Keep native rubber hands and make the tabletop visible to head RGB-D."""

    root = ET.fromstring(robot_xml)
    head_camera = root.find(".//camera[@name='head_camera']")
    if head_camera is None:
        raise ValueError("unitree_g1.xml has no head_camera")
    head_camera.set("xyaxes", "0 -1 0 0.207912 0 0.978148")

    for hand in ("left", "right"):
        wrist = root.find(f".//body[@name='{hand}_wrist_yaw_link']")
        if wrist is None:
            raise ValueError(f"unitree_g1.xml has no {hand} wrist body")
        native_visual = wrist.find(f"geom[@mesh='{hand}_rubber_hand']")
        native_collision = wrist.find(f"geom[@name='{hand}_hand_collision']")
        if native_visual is None or native_collision is None:
            raise ValueError(f"unitree_g1.xml has no native {hand} hand")
        native_collision.set("contype", "1")
        native_collision.set("conaffinity", "1")

    return ET.tostring(root, encoding="utf-8")


def install_g1_native_hand(model_module: Any) -> None:
    """Patch only camera/collision metadata; never add hand bodies or joints."""

    if getattr(model_module, "_luxi_native_hand", False):
        return
    original_get_assets = model_module.get_assets

    def get_assets_with_native_hand() -> dict[str, bytes]:
        assets = original_get_assets()
        robot_xml = assets.get("unitree_g1.xml")
        if robot_xml is None:
            raise RuntimeError("unitree_g1.xml asset is unavailable")
        assets["unitree_g1.xml"] = configure_g1_native_hand(robot_xml)
        return assets

    model_module.get_assets = get_assets_with_native_hand
    model_module._luxi_native_hand = True
