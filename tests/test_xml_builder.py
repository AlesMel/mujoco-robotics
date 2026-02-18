"""Unit tests for reach-scene XML assembly helpers."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from mujoco_robot.core.xml_builder import build_reach_xml
from mujoco_robot.tasks.lift_suction.lift_suction_env import URLiftSuctionEnv


_ROBOT_XML_NO_TABLE = """
<mujoco model="unit_robot">
  <worldbody>
    <body name="base" pos="-0.25 0 0.80">
      <body name="wrist3">
        <site name="ee_site" pos="0 0 0" quat="1 0 0 0"/>
        <geom name="ee_sphere" type="sphere" pos="0 0 0" size="0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()


_ROBOT_XML_WITH_TABLE = """
<mujoco model="unit_robot">
  <worldbody>
    <geom name="table" type="box" pos="0 0 0.72" size="0.6 0.6 0.02"/>
    <geom name="floor" type="plane" pos="0 0 0" size="4 4 0.1"/>
    <body name="base" pos="-0.30 0 0.74"/>
  </worldbody>
</mujoco>
""".strip()


def test_build_reach_xml_injects_table_and_floor_when_missing() -> None:
    """Custom robot XMLs without scene furniture should get table + floor."""
    model_xml = build_reach_xml(
        _ROBOT_XML_NO_TABLE,
        render_size=(320, 240),
        reach_threshold=0.06,
    )
    root = ET.fromstring(model_xml)

    table = root.find("./worldbody/geom[@name='table']")
    floor = root.find("./worldbody/geom[@name='floor']")
    assert table is not None
    assert floor is not None

    table_pos = [float(v) for v in table.get("pos", "").split()]
    table_size = [float(v) for v in table.get("size", "").split()]
    assert len(table_pos) == 3
    assert len(table_size) == 3
    assert table_size[0] == pytest.approx(0.45)
    assert table_size[1] == pytest.approx(0.55)

    # Base is at z=0.80, table top should land on that level.
    assert (table_pos[2] + table_size[2]) == pytest.approx(0.80)
    # Robot should be near the rear table edge (mount margin ~= 14 cm).
    rear_edge_x = table_pos[0] - table_size[0]
    assert (-0.25 - rear_edge_x) == pytest.approx(0.14, abs=1e-3)

    # IsaacLab-like furniture parts should be present.
    assert root.find("./worldbody/geom[@name='robot_pedestal']") is not None
    assert root.find("./worldbody/geom[@name='table_apron_n']") is not None
    assert root.find("./worldbody/geom[@name='table_leg_fl']") is not None
    table_top_mat = root.find("./asset/material[@name='scene_table_top_mat']")
    assert table_top_mat is not None
    assert table_top_mat.get("rgba") == "0.04 0.04 0.04 1"


def test_build_reach_xml_normalizes_existing_scene_furniture() -> None:
    """Scene assembly should replace old table/floor with canonical layout."""
    model_xml = build_reach_xml(
        _ROBOT_XML_WITH_TABLE,
        render_size=(320, 240),
        reach_threshold=0.06,
    )
    root = ET.fromstring(model_xml)

    assert len(root.findall("./worldbody/geom[@name='table']")) == 1
    assert len(root.findall("./worldbody/geom[@name='floor']")) == 1
    assert len(root.findall("./worldbody/geom[@name='table_leg_fl']")) == 1
    assert len(root.findall("./worldbody/geom[@name='robot_pedestal']")) == 1


def test_lift_suction_xml_uses_same_canonical_table_as_reach() -> None:
    """Lift-suction scene should use the same canonical table helper as reach."""
    env = URLiftSuctionEnv.__new__(URLiftSuctionEnv)
    env.render_size = (320, 240)
    model_xml = env._build_env_xml(_ROBOT_XML_NO_TABLE)
    root = ET.fromstring(model_xml)

    table = root.find("./worldbody/geom[@name='table']")
    floor = root.find("./worldbody/geom[@name='floor']")
    assert table is not None
    assert floor is not None

    table_pos = [float(v) for v in table.get("pos", "").split()]
    table_size = [float(v) for v in table.get("size", "").split()]
    assert table_size == pytest.approx([0.45, 0.55, 0.02], abs=1e-4)
    # Base is at z=0.80 in _ROBOT_XML_NO_TABLE so table top should match.
    assert (table_pos[2] + table_size[2]) == pytest.approx(0.80, abs=1e-4)

    # Canonical furniture pieces from ensure_spawn_table().
    assert root.find("./worldbody/geom[@name='robot_pedestal']") is not None
    assert root.find("./worldbody/geom[@name='table_apron_n']") is not None
    assert root.find("./worldbody/geom[@name='table_leg_fl']") is not None

    table_top_mat = root.find("./asset/material[@name='scene_table_top_mat']")
    assert table_top_mat is not None
    assert table_top_mat.get("rgba") == "0.04 0.04 0.04 1"
