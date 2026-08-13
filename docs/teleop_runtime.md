# Newton Thread Teleop Runtime

This is the real-time path for running the reconstructed thread/gripper scene on
a GPU computer and controlling it from a VR frontend such as Unity + Quest.

The current runtime is a stable v0:

- UDP command input
- UDP JSON state output
- reconstructed thread initialized from `newton_frame_scene.npz`
- PSM gripper pose initialized from the pose-estimation output
- kinematic endpoint grasp/drag after `grip=true`
- no USD/OBJ/Blender export in the runtime loop

It is not yet a true Newton friction/contact grasp. The grasp is currently a
hard-coded kinematic attachment so the VR/frontend integration can be built
without the cable solver exploding.

## Runtime Files

- `tools/teleop_kinematic_server.py`
  - GPU-machine runtime server
  - receives teleop commands
  - streams thread nodes and gripper state

- `tools/teleop_mock_client.py`
  - small Python client for testing without VR
  - sends a reach/grip/drag command sequence

- `scripts/run_teleop_kinematic_server_docker.sh`
  - Docker wrapper for the server

## Data Required On A New GPU Computer

The code repo does not include the large data files. Copy or recreate these on
the target machine:

```text
newton_frame_scene_000000/newton_frame_scene.npz
thread_with_gripper_pose_frames/episode_0000_inputs/joint_000000.npy
thread_with_gripper_pose_frames/episode_0000_inputs/jaw_000000.npy
assets/dvrk/psm1_si.urdf
assets/dvrk_model/meshes/...
```

The Blackdragon paths currently used by the runner are:

```text
~/thread_recon_run/thread_reconstruction-kf/newton_frame_scene_000000/newton_frame_scene.npz
~/thread_recon_run/thread_reconstruction-kf/thread_with_gripper_pose_frames/episode_0000_inputs/joint_000000.npy
~/thread_recon_run/assets/dvrk/psm1_si.urdf
~/thread_recon_run/assets/dvrk_model/meshes/...
```

Inside Docker, those resolve to:

```text
/workspace/thread_reconstruction/newton_frame_scene_000000/newton_frame_scene.npz
/workspace/thread_reconstruction/thread_with_gripper_pose_frames/episode_0000_inputs/joint_000000.npy
/workspace/thread_reconstruction/thread_with_gripper_pose_frames/episode_0000_inputs/jaw_000000.npy
/workspace/home/thread_recon_run/assets/dvrk/psm1_si.urdf
/workspace/home/thread_recon_run/assets/dvrk_model/meshes/...
```

## Run The Server

From the project root on the GPU machine:

```bash
COMMAND_PORT=8765 RATE_HZ=90 \
bash scripts/run_teleop_kinematic_server_docker.sh
```

The server listens on UDP port `8765`.

## Test Without VR

In a second terminal on the same machine:

```bash
python3 tools/teleop_mock_client.py \
  --host 127.0.0.1 \
  --port 8765 \
  --duration 6 \
  --rate-hz 60
```

Expected output:

```text
t=... grip=True disp=0.01...m update=...ms nodes=64
sent=... received=...
```

## UDP Command Protocol

Send UTF-8 JSON datagrams to the server.

Example command:

```json
{
  "type": "teleop",
  "seq": 12,
  "grip": true,
  "jaw": 0.02,
  "delta_newton": [0.018, 0.0, 0.012]
}
```

Fields:

- `target_newton`: absolute gripper target in Newton-view coordinates.
- `delta_newton`: offset from the selected thread-end home target.
- `grip`: when `true`, the selected thread endpoint is attached/dragged.
- `jaw`: open fraction from `0.0` closed to `1.0` open.
- `reset`: reset thread and robot state.

Use either `target_newton` or `delta_newton`. For initial VR work,
`delta_newton` is easier.

## UDP State Protocol

The server replies to the sender with JSON:

```json
{
  "type": "thread_teleop_state",
  "version": 1,
  "seq": 12,
  "time": 1.23,
  "target_thread_idx": 0,
  "target_newton": [0.04, 0.12, -0.02],
  "jaw_grasp_newton": [0.04, 0.12, -0.02],
  "grip": true,
  "jaw_open_fraction": 0.02,
  "target_displacement_m": 0.018,
  "thread_nodes_newton": [[...], "..."],
  "joint_values": {"yaw": 0.5, "...": 0.0},
  "perf": {"update_seconds": 0.002, "rate_hz": 90.0}
}
```

For Unity/Quest:

- render `thread_nodes_newton` as a tube/line mesh
- use `jaw_grasp_newton` for a simple gripper proxy initially
- later load the PSM mesh/URDF in Unity and drive it from `joint_values`

## Unity / Quest Plan

1. Create Unity OpenXR/Meta Quest project.
2. Add a UDP client component.
3. Read controller pose and trigger.
4. Convert controller motion into a scaled `delta_newton`.
5. Send `grip=true` while trigger is held.
6. Render streamed `thread_nodes_newton`.
7. Render a simple gripper proxy first.
8. Replace proxy with PSM mesh/URDF once the interaction feels right.

Recommended mapping:

```text
controller position delta * 0.1 -> delta_newton
trigger > 0.7 -> grip=true, jaw=0.02
trigger <= 0.7 -> grip=false, jaw=1.0
grip button -> clutch/recenter
```

## Next Step For True Physics

After the VR loop works, replace the kinematic endpoint drag with a proper
Newton attachment/contact model:

- simulator-native jaw collision bodies
- stable cable-node grasp constraint
- stiffness/damping tuned at a fixed timestep
- no teleporting partial cable bodies
