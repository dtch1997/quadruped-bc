import numpy as np
import mujoco
import mediapy as media



model = mujoco.MjModel.from_xml_path("unitree_a1/a1.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

# Parameters.
DURATION = 12         # seconds
FRAMERATE = 60        # Hz
TOTAL_ROTATION = 15   # degrees
CTRL_STD = 0.05       # actuator units
CTRL_RATE = 0.8       # seconds

# Make new camera, set distance.
camera = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, camera)
camera.distance = 2.3

# Enable contact force visualisation.
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

# Set the scale of visualized contact forces to 1cm/N.
model.vis.map.force = 0.01

# Define smooth orbiting function.
def unit_smooth(normalised_time: float) -> float:
    return 1 - np.cos(normalised_time*2*np.pi)
def azimuth(time: float) -> float:
    return 100 + unit_smooth(data.time/DURATION) * TOTAL_ROTATION

# Precompute some noise.
np.random.seed(1)
nsteps = int(np.ceil(DURATION/model.opt.timestep))
perturb = np.random.randn(nsteps, nu)

# Smooth the noise.
width = int(nsteps * CTRL_RATE/DURATION)
kernel = np.exp(-0.5*np.linspace(-3, 3, width)**2)
kernel /= np.linalg.norm(kernel)
for i in range(nu):
  perturb[:, i] = np.convolve(perturb[:, i], kernel, mode='same')

# Reset data, set initial pose
# TODO

# New renderer instance with higher resolution.
renderer = mujoco.Renderer(model, width=1280, height=720)

frames = []
while data.time < DURATION:
    # Set the action.
    data.ctrl[:] = np.random.randn(model.nu)

    # Step the simulation.
    mujoco.mj_step(model, data)

    # Render and save frames.
    if len(frames) < data.time * FRAMERATE:
        camera.azimuth = azimuth(data.time)
        renderer.update_scene(data, camera, scene_option)
        pixels = renderer.render()
        frames.append(pixels)
