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


def _parse_vec3(value: str | None) -> tuple[float, float, float] | None:
    """Parse a 3-float whitespace-separated vector string."""
    if value is None:
        return None
    parts = value.split()
    if len(parts) != 3:
        return None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None


def _ensure_material(
    root: ET.Element,
    name: str,
    rgba: str,
    specular: str = "0.1",
    shininess: str = "0.05",
) -> None:
    """Ensure a named material exists in ``<asset>``."""
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    if asset.find(f"./material[@name='{name}']") is not None:
        return
    ET.SubElement(asset, "material", {
        "name": name,
        "rgba": rgba,
        "specular": specular,
        "shininess": shininess,
    })


def _remove_world_geoms(
    worldbody: ET.Element,
    exact_names: set[str],
    prefix_names: tuple[str, ...],
) -> None:
    """Remove matching worldbody geoms in-place."""
    for geom in list(worldbody.findall("geom")):
        name = geom.get("name", "")
        if name in exact_names or any(name.startswith(p) for p in prefix_names):
            worldbody.remove(geom)


def ensure_spawn_table(
    root: ET.Element,
    table_half_xy: tuple[float, float] = (0.45, 0.55),
    table_half_thickness: float = 0.02,
    base_mount_margin_x: float = 0.14,
) -> None:
    """Build an IsaacLab-like table scene anchored to the robot base.

    This canonicalises world furniture for reach scenes:
    floor + tabletop + frame + legs + robot pedestal mount.
    Existing ``table`` / ``floor`` geoms are replaced so all robots
    share a consistent layout.
    """
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Robot MJCF missing <worldbody>")

    base_x, base_y, base_z = 0.0, 0.0, 0.74
    base_body = root.find(".//body[@name='base']")
    if base_body is not None:
        parsed = _parse_vec3(base_body.get("pos"))
        if parsed is not None:
            base_x, base_y, base_z = parsed

    # IsaacLab-like industrial palette with a black tabletop.
    _ensure_material(root, "scene_table_top_mat", "0.04 0.04 0.04 1")
    _ensure_material(root, "scene_table_leg_mat", "0.08 0.08 0.08 1", "0.05", "0.02")
    _ensure_material(root, "scene_floor_mat", "0.87 0.88 0.90 1", "0.0", "0.0")

    _remove_world_geoms(
        worldbody,
        exact_names={"table", "floor", "robot_pedestal"},
        prefix_names=("table_leg_", "table_apron_"),
    )

    half_x, half_y = float(table_half_xy[0]), float(table_half_xy[1])
    half_z = max(0.001, float(table_half_thickness))
    margin_x_max = max(0.05, half_x - 0.02)
    margin_x = min(max(float(base_mount_margin_x), 0.05), margin_x_max)

    # Keep the robot near the rear edge, as in typical bench setups.
    center_x = base_x + half_x - margin_x
    center_y = base_y
    top_z = base_z
    center_z = max(half_z, top_z - half_z)
    under_top_z = center_z - half_z

    ET.SubElement(worldbody, "geom", {
        "name": "floor",
        "type": "plane",
        "pos": "0 0 0",
        "size": "5 5 0.1",
        "material": "scene_floor_mat",
        "friction": "1 0.1 0.1",
    })

    table_attrs = {
        "name": "table",
        "type": "box",
        "pos": f"{center_x:.4f} {center_y:.4f} {center_z:.4f}",
        "size": f"{half_x:.4f} {half_y:.4f} {half_z:.4f}",
        "material": "scene_table_top_mat",
        "friction": "1 0.1 0.1",
    }
    ET.SubElement(worldbody, "geom", table_attrs)

    apron_half_t = 0.02
    apron_half_h = 0.03
    apron_z = max(apron_half_h, under_top_z - apron_half_h)

    ET.SubElement(worldbody, "geom", {
        "name": "table_apron_n",
        "type": "box",
        "pos": f"{center_x:.4f} {center_y + (half_y - apron_half_t):.4f} {apron_z:.4f}",
        "size": f"{half_x:.4f} {apron_half_t:.4f} {apron_half_h:.4f}",
        "material": "scene_table_leg_mat",
        "contype": "0",
        "conaffinity": "0",
        "mass": "0",
    })
    ET.SubElement(worldbody, "geom", {
        "name": "table_apron_s",
        "type": "box",
        "pos": f"{center_x:.4f} {center_y - (half_y - apron_half_t):.4f} {apron_z:.4f}",
        "size": f"{half_x:.4f} {apron_half_t:.4f} {apron_half_h:.4f}",
        "material": "scene_table_leg_mat",
        "contype": "0",
        "conaffinity": "0",
        "mass": "0",
    })
    ET.SubElement(worldbody, "geom", {
        "name": "table_apron_e",
        "type": "box",
        "pos": f"{center_x + (half_x - apron_half_t):.4f} {center_y:.4f} {apron_z:.4f}",
        "size": f"{apron_half_t:.4f} {half_y:.4f} {apron_half_h:.4f}",
        "material": "scene_table_leg_mat",
        "contype": "0",
        "conaffinity": "0",
        "mass": "0",
    })
    ET.SubElement(worldbody, "geom", {
        "name": "table_apron_w",
        "type": "box",
        "pos": f"{center_x - (half_x - apron_half_t):.4f} {center_y:.4f} {apron_z:.4f}",
        "size": f"{apron_half_t:.4f} {half_y:.4f} {apron_half_h:.4f}",
        "material": "scene_table_leg_mat",
        "contype": "0",
        "conaffinity": "0",
        "mass": "0",
    })

    leg_half_xy = 0.03
    leg_top_z = max(0.2, apron_z - apron_half_h)
    leg_half_z = max(0.1, leg_top_z / 2.0)
    leg_center_z = leg_half_z
    leg_off_x = max(leg_half_xy + 0.02, half_x - leg_half_xy - 0.05)
    leg_off_y = max(leg_half_xy + 0.02, half_y - leg_half_xy - 0.05)

    for suffix, sx, sy in (
        ("fl", +1.0, +1.0),
        ("fr", +1.0, -1.0),
        ("rl", -1.0, +1.0),
        ("rr", -1.0, -1.0),
    ):
        ET.SubElement(worldbody, "geom", {
            "name": f"table_leg_{suffix}",
            "type": "box",
            "pos": (
                f"{center_x + sx * leg_off_x:.4f} "
                f"{center_y + sy * leg_off_y:.4f} "
                f"{leg_center_z:.4f}"
            ),
            "size": f"{leg_half_xy:.4f} {leg_half_xy:.4f} {leg_half_z:.4f}",
            "material": "scene_table_leg_mat",
            "contype": "0",
            "conaffinity": "0",
            "mass": "0",
        })

    # Visual pedestal to indicate where the arm is mounted on the table.
    ped_half_z = 0.04
    ped_center_z = max(ped_half_z, top_z - ped_half_z)
    ET.SubElement(worldbody, "geom", {
        "name": "robot_pedestal",
        "type": "cylinder",
        "pos": f"{base_x:.4f} {base_y:.4f} {ped_center_z:.4f}",
        "size": "0.085 0.040",
        "material": "scene_table_leg_mat",
        "contype": "0",
        "conaffinity": "0",
        "mass": "0",
    })


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

    # Read the ee_site's local quaternion so the visual axes match the
    # frame that is actually measured by site_xmat.  Without this the
    # RGB axes show the raw wrist3 body orientation which differs from
    # the ee_site orientation by the site's local rotation — making it
    # look like orientation tracking is broken.
    ee_site_quat = None
    for site in wrist3.findall("site"):
        if site.get("name") == "ee_site":
            ee_site_quat = site.get("quat")
            break

    ee_frame_attrs = {
        "name": "ee_frame",
        "pos": ee_pos,
    }
    if ee_site_quat is not None:
        ee_frame_attrs["quat"] = ee_site_quat

    ee_frame = ET.SubElement(wrist3, "body", ee_frame_attrs)
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
        3. Ensure spawn table + floor.
        4. Inject goal marker (cube + RGB axes).
        5. Inject EE coordinate axes.
        6. Inject side camera.

    Returns the modified XML as a string.
    """
    root = ET.fromstring(robot_xml)
    set_framebuffer_size(root, render_size[0], render_size[1])
    ensure_spawn_table(root)
    inject_goal_marker(root, reach_threshold)
    inject_ee_axes(root, reach_threshold)
    inject_side_camera(root)
    return ET.tostring(root, encoding="unicode")
