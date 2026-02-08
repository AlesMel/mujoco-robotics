"""XML-level helpers for injecting elements into robot MJCF files.

These functions operate on ``xml.etree.ElementTree`` objects so the
environment classes don't have to know the raw XML layout.
"""
from __future__ import annotations

import math
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


def _add_axes_frame(
    parent: ET.Element,
    prefix: str,
    axis_len: float,
    axis_rad: float,
    alpha: float = 0.85,
    ghost_yz: bool = False,
) -> None:
    """Add RGB coordinate-frame axes (X=red, Y=green, Z=blue) to *parent*.

    Each axis is a thin cylinder drawn from the origin along its direction.
    ``prefix`` is prepended to geom names to keep them unique.

    If *ghost_yz* is True, the Y and Z axes are drawn very faintly
    (useful for the goal marker where only yaw/heading is controlled).
    """
    yz_alpha = 0.12 if ghost_yz else alpha
    axes = [
        ("x", f"{axis_len} 0 0", f"1 0 0 {alpha}"),
        ("y", f"0 {axis_len} 0", f"0 1 0 {yz_alpha}"),
        ("z", f"0 0 {axis_len}", f"0 0 1 {yz_alpha}"),
    ]
    for axis_name, endpoint, rgba in axes:
        ET.SubElement(parent, "geom", {
            "name": f"{prefix}_axis_{axis_name}",
            "type": "cylinder",
            "fromto": f"0 0 0 {endpoint}",
            "size": str(round(axis_rad, 5)),
            "rgba": rgba,
            "contype": "0",
            "conaffinity": "0",
            "mass": "0",
        })


def inject_goal_marker(
    root: ET.Element,
    reach_threshold: float,
    initial_pos: str = "0.1 0 0.95",
) -> None:
    """Add a goal marker with a translucent cube and RGB coordinate axes.

    The cube shows the target position; the RGB axes (X=red, Y=green,
    Z=blue) show the desired full 3-D orientation.  The goal body's
    quaternion is set at runtime to the sampled target orientation.
    """
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Robot MJCF missing <worldbody>")

    goal_body = ET.SubElement(worldbody, "body", {
        "name": "goal_body",
        "pos": initial_pos,
    })
    # Invisible site used for distance calculations
    ET.SubElement(goal_body, "site", {
        "name": "goal_site",
        "size": "0.01",
        "rgba": "1 0.2 0.2 0.0",
    })
    # Translucent cube — shows the target position
    cube_half = round(reach_threshold * 0.35, 4)
    ET.SubElement(goal_body, "geom", {
        "name": "goal_cube",
        "type": "box",
        "size": f"{cube_half} {cube_half} {cube_half}",
        "rgba": "0.9 0.15 0.15 0.25",
        "contype": "0",
        "conaffinity": "0",
        "mass": "0",
    })
    # RGB coordinate axes — all 3 axes shown at full alpha
    # since this task controls full 3-D orientation.
    axis_len = round(reach_threshold * 2.5, 4)
    axis_rad = round(reach_threshold * 0.12, 4)
    _add_axes_frame(goal_body, "goal", axis_len, axis_rad, alpha=0.8)


def inject_ee_axes(
    root: ET.Element,
    reach_threshold: float,
) -> None:
    """Replace the ``ee_sphere`` geom with RGB coordinate axes on the EE.

    The axes are placed inside a child body of ``wrist3`` at the
    ``ee_site`` position/orientation so they rotate with the tool flange.
    """
    # Find ee_sphere and its parent (wrist3 body)
    wrist3 = None
    ee_sphere = None
    for body in root.iter("body"):
        for geom in body.findall("geom"):
            if geom.get("name") == "ee_sphere":
                wrist3 = body
                ee_sphere = geom
                break
        if ee_sphere is not None:
            break

    if wrist3 is None or ee_sphere is None:
        return  # nothing to do — no ee_sphere found

    # Read position from the existing ee_sphere
    ee_pos = ee_sphere.get("pos", "0 0 0")

    # Remove the old sphere
    wrist3.remove(ee_sphere)

    # Add a child body at the ee_sphere position with RGB axes.
    # No local rotation — the axes show the raw tool-flange frame
    # so they match the goal axes when the full orientation is aligned.
    ee_frame = ET.SubElement(wrist3, "body", {
        "name": "ee_frame",
        "pos": ee_pos,
    })
    axis_len = round(reach_threshold * 2.5, 4)
    axis_rad = round(reach_threshold * 0.12, 4)
    _add_axes_frame(ee_frame, "ee", axis_len, axis_rad, alpha=0.7)


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
        3. Inject goal marker (cube + RGB axes).
        4. Inject EE coordinate axes.
        5. Inject side camera.

    Returns the modified XML as a string.
    """
    root = ET.fromstring(robot_xml)
    set_framebuffer_size(root, render_size[0], render_size[1])
    inject_goal_marker(root, reach_threshold)
    inject_ee_axes(root, reach_threshold)
    inject_side_camera(root)
    return ET.tostring(root, encoding="unicode")
