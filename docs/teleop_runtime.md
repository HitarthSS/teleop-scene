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

- `tools/keyboard_teleop_client.py`
  - host-side interactive keyboard controller
  - sends `delta_newton`, `jaw`, and `grip` commands to the UDP server

- `tools/live_thread_viewer.py`
  - host-side live matplotlib viewer
  - subscribes to the UDP state stream and renders the thread/gripper point

- `tools/live_thread_newton_viewer.py`
  - Newton ViewerGL live viewer
  - runs best inside the Docker image with X11/OpenGL access

- `tools/oculus_reader_teleop_client.py`
  - host-side bridge from `rail-berkeley/oculus_reader` to the UDP server
  - reads Quest controller transforms/buttons
  - sends `delta_newton`, `jaw`, and `grip` commands

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

## Use Keyboard Input

Keep the Newton teleop server running in one terminal. In a second terminal on
the same computer:

```bash
cd ~/Desktop/teleop-scene

python3 tools/keyboard_teleop_client.py \
  --server-host 127.0.0.1 \
  --server-port 8765
```

Controls:

```text
a/d      x -/+
w/s      y +/-
q/e      z +/-
space    toggle grasp / close jaw
r        reset/recenter
x        zero motion delta
+/-      increase/decrease step size
Esc      quit
```

The client sends the same UDP command protocol as Oculus/Unity:

```json
{"grip": true, "jaw": 0.02, "delta_newton": [0.01, 0.0, 0.005]}
```

## See It Live

Use three terminals:

Terminal 1: run the Docker/Newton server.

Terminal 2: run keyboard control.

```bash
cd ~/Desktop/teleop-scene
python3 tools/keyboard_teleop_client.py --server-host 127.0.0.1 --server-port 8765
```

Terminal 3: run the live viewer.

```bash
cd ~/Desktop/teleop-scene
python3 tools/live_thread_viewer.py --server-host 127.0.0.1 --server-port 8765
```

If matplotlib is missing:

```bash
python3 -m pip install matplotlib numpy
```

The viewer renders:

- blue line: thread
- cyan/yellow points: thread start/end
- magenta x: commanded target
- black point: gripper grasp point
- red line: grabbed-end trail

## See It Live With Newton ViewerGL

This uses Newton's own OpenGL viewer instead of matplotlib. It needs an X11
desktop session and a Docker image rebuilt after the ViewerGL dependencies were
added.

Rebuild the image after pulling the latest code:

```bash
cd ~/Desktop/teleop-scene
sudo docker build -t thread-reconstruction-newton:latest -f docker/Dockerfile.newton .
```

Allow local Docker windows to connect to X11:

```bash
xhost +local:docker
```

Use three terminals:

Terminal 1: run the Docker/Newton teleop server.

Terminal 2: run keyboard control.

```bash
cd ~/Desktop/teleop-scene
python3 tools/keyboard_teleop_client.py --server-host 127.0.0.1 --server-port 8765
```

Terminal 3: run Newton ViewerGL.

On a normal NVIDIA Docker setup:

```bash
cd ~/Desktop/teleop-scene
bash scripts/run_live_thread_newton_viewer_docker.sh
```

On the Alienware machine that rejected `--gpus all` and requested
`--runtime=nvidia`:

```bash
cd ~/Desktop/teleop-scene
DOCKER_GPU_ARGS="--runtime=nvidia" \
bash scripts/run_live_thread_newton_viewer_docker.sh
```

If X11 blocks the window, rerun:

```bash
xhost +local:docker
```

## Use OculusReader Instead Of Unity

This path uses the Berkeley OculusReader project:

```text
https://github.com/rail-berkeley/oculus_reader
```

OculusReader must run on the host computer, not inside the Newton Docker
container, because it needs ADB access to the Quest headset.

Install host dependencies:

```bash
sudo apt-get update
sudo apt-get install -y android-tools-adb
python3 -m pip install git+https://github.com/rail-berkeley/oculus_reader.git
```

The upstream README says active Quest 3 support has moved to this fork:

```bash
python3 -m pip install git+https://github.com/jborbik/oculus_reader.git
```

Use one or the other. Start with the `rail-berkeley` package; if the Quest 3
does not stream correctly, switch to the `jborbik` fork.

Quest setup:

1. Enable Developer Mode in the Meta Quest mobile app.
2. Connect the headset to the computer with USB-C.
3. Put on the headset and accept USB debugging.
4. Verify ADB:

```bash
adb devices
```

You should see a device listed as `device`, not `unauthorized`.

Keep the Newton teleop server running in one terminal. In a second terminal,
run:

```bash
cd ~/Desktop/teleop-scene

python3 tools/oculus_reader_teleop_client.py \
  --server-host 127.0.0.1 \
  --server-port 8765 \
  --hand right \
  --rate-hz 60 \
  --position-scale 0.10
```

Controls:

- hold controller grip: move/clutch the simulated gripper
- trigger: close/grasp the thread
- A/B/right joystick press: recenter controller origin

If the movement direction feels wrong, change the axis map:

```bash
python3 tools/oculus_reader_teleop_client.py --axis-map x,z,-y
python3 tools/oculus_reader_teleop_client.py --axis-map x,-z,y
python3 tools/oculus_reader_teleop_client.py --axis-map -x,z,-y
```

Wireless ADB option after USB setup:

```bash
adb shell ip route
adb tcpip 5555
```

Find the Quest IP after `src`, then run:

```bash
python3 tools/oculus_reader_teleop_client.py --oculus-ip QUEST_IP_ADDRESS
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
