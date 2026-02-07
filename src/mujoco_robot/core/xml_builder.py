"""XML-level helpers for injecting elements into robot MJCF files.

These functions operate on ``xml.etree.ElementTree`` objects so the
environment classes don't have to know the raw XML layout.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple


def load_robot_xml(path: str) -> str:
    """Read a robot MJCF file from disk.

    If the ``<compiler>`` tag contains a *relative* ``meshdir``, it is
    resolved to an absolute path so that ``mujoco.MjModel.from_xml_string``
    can locate mesh assets regardless of the current working directory.

    Raises ``FileNotFoundError`` with a helpful message if missing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Robot MJCF not found at '{path}'.  "
            "Make sure the XML file exists."
        )
    xml_text = p.read_text()

    # Resolve relative meshdir → absolute so from_xml_string works
    root = ET.fromstring(xml_text)
    compiler = root.find("compiler")
    if compiler is not None:
        meshdir = compiler.get("meshdir")
        if meshdir and not Path(meshdir).is_absolute():
            abs_meshdir = str((p.parent / meshdir).resolve())
            compiler.set("meshdir", abs_meshdir)
            xml_text = ET.tostring(root, encoding="unicode")

    return xml_text


def set_framebuffer_size(
    root: ET.Element, width: int, height: int
) -> None:
    """Ensure the offscreen framebuffer is at least ``width x height``."""
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    gl = visual.find("global")
    if gl is None:
        gl = ET.SubElement(visual, "global")
    gl.set("offwidth", str(max(width, 640)))
    gl.set("offheight", str(max(height, 480)))


def inject_goal_marker(
    root: ET.Element,
    reach_threshold: float,
    initial_pos: str = "0.1 0 0.95",
) -> None:
    """Add a translucent goal sphere + invisible site to the worldbody."""
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Robot MJCF missing <worldbody>")

    goal_body = ET.SubElement(worldbody, "body", {
        "name": "goal_body",
        "pos": initial_pos,
    })
    ET.SubElement(goal_body, "site", {
        "name": "goal_site",
        "size": "0.01",
        "rgba": "1 0.2 0.2 0.0",
    })
    ET.SubElement(goal_body, "geom", {
        "name": "goal_sphere",
        "type": "sphere",
        "size": str(reach_threshold),
        "rgba": "0.9 0.15 0.15 0.35",
        "contype": "0",
        "conaffinity": "0",
        "mass": "0",
    })


def inject_side_camera(root: ET.Element) -> None:
    """Add a second camera for dual-view rendering."""
    worldbody = root.find("worldbody")
    if worldbody is None:
        return
    ET.SubElement(worldbody, "camera", {
        "name": "side",
        "pos": "1.2 -1.0 1.6",
        "xyaxes": "0.6 0.8 0 -0.3 0.2 0.9",
        "mode": "fixed",
    })


def build_reach_xml(
    robot_xml: str,
    render_size: Tuple[int, int],
    reach_threshold: float,
) -> str:
    """Assemble the full reach-task MJCF from a base robot XML string.

    Steps:
        1. Parse the robot MJCF.
        2. Set framebuffer size.
        3. Inject goal marker.
        4. Inject side camera.

    Returns the modified XML as a string.
    """
    root = ET.fromstring(robot_xml)
    set_framebuffer_size(root, render_size[0], render_size[1])
    inject_goal_marker(root, reach_threshold)
    inject_side_camera(root)
    return ET.tostring(root, encoding="unicode")
