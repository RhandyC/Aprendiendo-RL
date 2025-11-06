import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN
from matplotlib import pyplot as plt

import highway_env  # noqa: F401

# import gymnasium as gym
import my_envs  # importante para que se registre


TRAIN = True

if __name__ == "__main__":
    # Create the environment
    # env = gym.make("highway-fast-v0", render_mode="rgb_array")

    # An example of customized env
    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 2,
            "features": ["presence", "x", "y"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            "absolute": False,
            "normalize": False,
            "clip": False,
            "order": "sorted"
        },
        "vehicles_count": 5,
        "initial_lane_id" : 1,
        "other_vehicles_type": "models.linear_idm_vehicle.LinearIDMVehicle",
        "duration" : 60,
        "lanes_count": 2
    }
    env = gym.make('my-highway-v0', config=config, render_mode='rgb_array')

    env = RecordVideo(
        env, video_folder="record_test/videos", episode_trigger=lambda e: True
    )
    env.unwrapped.set_record_video_wrapper(env)

    obs, info = env.reset()
    done = truncated = False
    i=0

    while not (done or truncated):
        # Predict
        action = 1
        # Get reward
        obs, reward, done, truncated, info = env.step(action)
        print("Observation" + str(i))
        print(obs)
        i=i+1
        # Render
        env.render()

    env.close()
    plt.imshow(env.render())
    plt.show()

