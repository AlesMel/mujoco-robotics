# MuJoCo Physics Simulation — How the Virtual Robot Works

> **Where this is used in our code:**
> [`xml_builder.py`](../src/mujoco_robot/core/xml_builder.py) — building the simulation scene
> [`configs.py`](../src/mujoco_robot/robots/configs.py) — robot parameters
> [`reach_env.py`](../src/mujoco_robot/envs/reach_env.py) — stepping the physics
> [`collision.py`](../src/mujoco_robot/core/collision.py) — self-collision detection

---

## Table of Contents

1. [What Is MuJoCo?](#1-what-is-mujoco)
2. [MJCF — The Robot Description Language](#2-mjcf--the-robot-description-language)
3. [Time, Steps, and Substeps](#3-time-steps-and-substeps)
4. [Actuators — How the Robot Moves](#4-actuators--how-the-robot-moves)
5. [Contacts & Collisions](#5-contacts--collisions)
6. [The Simulation Loop](#6-the-simulation-loop)
7. [Our Scene: XML Injection](#7-our-scene-xml-injection)
8. [Robot Configuration](#8-robot-configuration)

---

## 1. What Is MuJoCo?

**MuJoCo** (Multi-Joint dynamics with Contact) is a physics engine designed for simulating robots, biomechanics, and anything that involves articulated rigid bodies in contact with their environment.

Think of it as a "virtual physics lab" where you can:
- Build robots from rigid parts connected by joints
- Simulate gravity, friction, motors, and collisions
- Run thousands of times faster than real-time

MuJoCo is the #1 choice for robotics RL because:
- **Fast**: hundreds of simulation steps per millisecond
- **Accurate**: proper contact dynamics (not just penetration detection)
- **Differentiable**: smooth physics (no discontinuities)
- **Free**: open-source since 2022 (previously required a license)

### Model and Data

MuJoCo separates the simulation into two objects:

```python
model = mujoco.MjModel.from_xml_string(xml)  # STATIC: geometry, masses, joint types
data = mujoco.MjData(model)                   # DYNAMIC: positions, velocities, forces
```

- **MjModel**: Everything that doesn't change during simulation (robot structure, joint limits, actuator gains, material properties). Created once from an XML file.
- **MjData**: Everything that changes each step (joint angles, joint velocities, contact forces, sensor readings). Updated by `mj_step()`.

---

## 2. MJCF — The Robot Description Language

MuJoCo robots are described in **MJCF** (MuJoCo XML Format). Here's a simplified view:

```xml
<mujoco>
  <!-- Compilation settings -->
  <compiler angle="radian" meshdir="assets/"/>
  
  <!-- Physics options -->
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <!-- Visual settings -->
  <visual>
    <global offwidth="960" offheight="720"/>
  </visual>

  <!-- Materials, textures -->
  <asset>
    <mesh name="link1" file="link1.stl"/>
    <material name="metal" rgba="0.6 0.6 0.6 1"/>
  </asset>

  <!-- The world -->
  <worldbody>
    <!-- Table (static) -->
    <body name="table">
      <geom type="box" size="0.5 0.5 0.02" pos="0 0 0.73"/>
    </body>

    <!-- Robot arm (articulated) -->
    <body name="base" pos="-0.3 0 0.74">
      <joint name="shoulder_pan" type="hinge" axis="0 0 1"/>
      <geom type="cylinder" size="0.06 0.05"/>
      
      <body name="upper_arm" pos="0 0 0.163">
        <joint name="shoulder_lift" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0  0.425 0 0"/>
        
        <!-- ... more links ... -->
      </body>
    </body>
  </worldbody>

  <!-- Motors -->
  <actuator>
    <position name="shoulder_pan_motor" joint="shoulder_pan" kp="400"/>
    <position name="shoulder_lift_motor" joint="shoulder_lift" kp="400"/>
  </actuator>
</mujoco>
```

### Key MJCF elements

| Element | Purpose | Example |
|---------|---------|---------|
| `<body>` | A rigid link with position/orientation | `<body name="upper_arm" pos="0 0 0.163">` |
| `<joint>` | Connects bodies with constrained motion | `<joint type="hinge" axis="0 0 1"/>` (revolute around Z) |
| `<geom>` | Collision shape + visual shape | `<geom type="capsule" size="0.04" fromto="0 0 0  0.425 0 0"/>` |
| `<site>` | A reference point (no physics) | `<site name="ee_site" pos="0 0 0.1"/>` (our EE target) |
| `<actuator>` | Motor that applies forces/torques | `<position joint="elbow" kp="400"/>` |
| `<camera>` | Viewpoint for rendering | `<camera name="top" pos="0 0 2" xyaxes="..."/>` |

### Body hierarchy = kinematic tree

Bodies are nested:
```
worldbody
  └── base
      └── shoulder_link
          └── upper_arm
              └── forearm
                  └── wrist1
                      └── wrist2
                          └── wrist3 (EE is here)
```

Each body inherits the position/orientation of its parent. When the shoulder rotates, **everything downstream** moves with it. This is the kinematic chain that FK traverses (see [guide 02](02_jacobian_and_inverse_kinematics.md)).

---

## 3. Time, Steps, and Substeps

### The timestep

```python
model.opt.timestep = 0.002  # 2 milliseconds per physics step
```

MuJoCo advances time in fixed increments. Smaller timestep = more accurate physics, but more computation.

### Substeps — the key concept

We don't call the RL policy at every physics step. Instead:

```
1 RL step = 16 physics substeps × 0.002s = 0.032s = 32ms
```

This gives us a **control rate of ~31 Hz** (about 31 policy decisions per second).

```
    RL step 1                    RL step 2
    │                            │
    ▼                            ▼
    ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐┌─┬─ ...
    │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ ││ │
    └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘└─┴─ ...
     ↑  2ms each = 16 substeps = 32ms
```

**Why substeps?**
- Physics needs small timesteps for stability (2ms)
- But the RL policy only needs to decide every ~30ms
- This matches real UR robots which accept commands at 10-125 Hz
- Also matches Isaac Lab's `decimation=2` at 60 Hz sim → ~30 Hz control

### In code

```python
# Set actuator targets (from the RL action)
for k, act_id in enumerate(robot_actuators):
    data.ctrl[act_id] = qpos_targets[k]

# Run 16 physics substeps with the same actuator targets
for _ in range(16):
    mujoco.mj_step(model, data)
```

---

## 4. Actuators — How the Robot Moves

### Position servos

Our robot uses **position actuators** — you tell each joint WHERE to go, and the actuator drives it there:

```xml
<position name="shoulder_pan_motor" joint="shoulder_pan" kp="400"/>
```

Internally, MuJoCo applies a force proportional to the error:

$$
\tau = k_p \cdot (q_{target} - q_{actual})
$$

This is a **P controller** (proportional controller). With $k_p = 400$ N·m/rad, a 0.01 rad error produces 4 N·m of torque.

### How we command the actuators

```python
# data.ctrl[i] is the position target for actuator i
# The actuator's internal PD controller handles the rest
data.ctrl[shoulder_pan_motor] = -3.14  # target: -π radians
```

### Why position servos?

Real UR robots have built-in joint controllers — you send position commands, and the robot handles torque-level control internally. Position servos in MuJoCo simulate this behaviour. The RL policy only needs to think about "where should each joint go?" not "how much torque to apply."

### Control range

Each actuator has limits matching the joint range:

```python
# Set actuator limits to match joint limits
for act in robot_actuators:
    model.actuator_ctrlrange[act] = [lo, hi]  # e.g., [-2π, 2π]
```

MuJoCo clamps `data.ctrl[act]` to this range, preventing impossible commands.

### Gain tuning

```python
model.actuator_gainprm[act, 0] = 400.0  # kp — position gain
```

If the gain is too low, the robot is "floppy" and can't track fast commands. If too high, it becomes stiff but may oscillate. 400 N·m/rad works well for our UR simulation.

---

## 5. Contacts & Collisions

### How MuJoCo handles contacts

Every physics step, MuJoCo:
1. **Detects** all overlapping geometry pairs (broad phase + narrow phase)
2. **Computes** contact forces that prevent interpenetration
3. **Applies** friction at contact points

This is what makes the robot's links not pass through each other (or through the table).

### Self-collision detection

Our `CollisionDetector` class checks for collisions between the robot's own links:

```python
class CollisionDetector:
    def __init__(self, model):
        # Find all robot geometries (suffixed with "_col")
        # Build pairs of non-adjacent links
        # Adjacent links (connected by a joint) are excluded
    
    def count(self, data):
        # For each contact in data.contact:
        #   If both geoms belong to non-adjacent robot links → collision!
        return collision_count
```

**Why exclude adjacent links?** Neighbouring links in a robot always touch at their joint — that's not a "collision," it's normal. We only care about non-adjacent links hitting each other (e.g., the wrist hitting the shoulder).

### The contype/conaffinity system

MuJoCo uses bitmask filtering to control which geoms can collide:

```xml
<!-- This geom DOES collide with other geoms -->
<geom name="arm_link" contype="1" conaffinity="1"/>

<!-- This geom is visual-only, NO collision -->
<geom name="goal_cube" contype="0" conaffinity="0"/>
```

Our goal marker and coordinate axes have `contype="0"` and `conaffinity="0"` so they don't interact with the physics — they're purely visual.

---

## 6. The Simulation Loop

Here's what MuJoCo does internally each time you call `mj_step()`:

```
    mj_step(model, data)
    │
    ├── 1. Forward kinematics
    │       Compute all body positions/orientations from joint angles
    │
    ├── 2. Collision detection
    │       Find all geometry pair overlaps
    │
    ├── 3. Compute forces
    │       ├── Gravity (9.81 m/s² downward)
    │       ├── Actuator forces (PD controller τ = kp·(target - actual))
    │       ├── Contact forces (prevent interpenetration)
    │       ├── Damping forces (-b·velocity)
    │       └── Friction (at contacts and joints)
    │
    ├── 4. Solve equations of motion
    │       M·q̈ = τ_total  →  q̈ = M⁻¹·τ_total
    │       (M is the mass matrix, τ_total is sum of all forces)
    │
    └── 5. Integrate
            q_vel += q̈ · dt      (update velocities)
            q_pos += q_vel · dt   (update positions)
            time  += dt
```

Our environment uses `mjtIntegrator.mjINT_IMPLICITFAST` — an implicit integrator that is more stable than explicit Euler for stiff systems like robot joints with high-gain PD controllers.

---

## 7. Our Scene: XML Injection

We don't write the entire MJCF from scratch. Instead, we start with a pre-made robot MJCF (from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)) and **inject** additional elements:

```python
def build_reach_xml(robot_xml, render_size, reach_threshold):
    root = ET.fromstring(robot_xml)       # Parse existing robot XML
    set_framebuffer_size(root, ...)       # Set render resolution
    inject_goal_marker(root, ...)         # Add goal cube + RGB axes
    inject_ee_axes(root, ...)             # Replace EE sphere with RGB axes
    inject_side_camera(root)              # Add side-view camera
    return ET.tostring(root)              # Serialize back to XML string
```

### Goal marker

A translucent red cube with RGB coordinate axes showing the target pose:
- **Red axis** (X) = right direction of the target
- **Green axis** (Y) = forward direction of the target
- **Blue axis** (Z) = up direction of the target

The goal body's position and quaternion are updated at runtime when goals are sampled.

### EE coordinate axes

The end-effector also has RGB axes, drawn from the raw tool-flange frame. When the EE orientation matches the goal orientation, the EE's RGB axes align perfectly with the goal's RGB axes.

---

## 8. Robot Configuration

Each supported robot has a configuration in `configs.py`:

```python
RobotConfig(
    name="ur5e",
    model_path="robots/ur5e.xml",
    base_pos=[-0.30, 0.0, 0.74],        # Base mounted on table
    link_lengths=[0.1625, 0.425, ...],   # From official UR DH parameters
    init_q=[-π, -π/2, π/2, -π/2, -π/2, 0],  # Home pose
    goal_bounds=[[-0.10, 0.55], ...],    # Where goals can be sampled
    ee_bounds=[[-1.10, 0.60], ...],      # Hard EE position clamps
    goal_distance=(0.25, 0.80),          # Min/max distance from base
    goal_min_height=0.85,                # Above table top
    goal_min_ee_dist=0.15,               # Not trivially close to EE
)
```

### Home pose

The `init_q` values define the robot's starting joint angles:

```python
init_q = [-π, -π/2, π/2, -π/2, -π/2, 0]
```

This is the "Menagerie-style home" — arm pointing up, forearm horizontal, wrist down. It's a good starting pose because:
- It's far from joint limits
- It's far from singularities
- The EE is in a reachable, well-conditioned region

### Goal sampling constraints

Goals are sampled randomly within `goal_bounds`, but must also satisfy:
- **Distance from base**: between `goal_distance[0]` and `goal_distance[1]` metres
- **Height**: above `goal_min_height` (prevents goals on/below the table)
- **Distance from EE**: at least `goal_min_ee_dist` (prevents trivially easy starts)

This ensures goals are **reachable but not trivial**.

### EE bounds

Hard clamps on the end-effector position. Even if the IK controller wants to send the EE somewhere wild, it gets clamped to these bounds. This prevents the robot from trying to reach behind its own base or below the table.

---

## Key Numbers to Remember

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `timestep` | 0.002s (2ms) | Physics step size |
| `n_substeps` | 16 | Physics steps per RL step |
| `control_dt` | 0.032s (32ms) | Time between RL decisions |
| `control_rate` | ~31 Hz | RL decisions per second |
| `kp` (actuator gain) | 400 N·m/rad | Stiffness of position servos |
| `damping` | 2.0 N·m·s/rad | Joint viscous friction |
| `gravity` | 9.81 m/s² | Standard Earth gravity |
| `settle_steps` | 300 | Physics steps to settle at reset |

---

**Previous:** [03 — The RL Environment](03_rl_environment.md)
**Next:** [05 — PPO Training](05_ppo_training.md) — how the neural network actually learns from experience.
