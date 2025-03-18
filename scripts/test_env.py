import os
import hydra
import datetime
import torch
from omegaconf import DictConfig
from omni.isaac.kit import SimulationApp
from ppo_new import PPO
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torchrl.transforms import VelController
from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
from torchrl.envs.transforms import TransformedEnv, Compose
from utils import evaluate
from torchrl.envs.utils import ExplorationType

FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")

@hydra.main(config_path=FILE_PATH, config_name="test", version_base=None)
def main(cfg: DictConfig):
    # Simulation App
    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})

    # Navigation Training Environment
    from env_new import NavigationEnv
    env = NavigationEnv(cfg)

    # Transformed Environment
    transforms = []
    controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=False)
    transforms.append(vel_transform)
    transformed_env = TransformedEnv(env, Compose(*transforms)).train()
    transformed_env.set_seed(cfg.seed)

    # PPO Policy
    policy = PPO(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)

    # checkpoint = "/home/pjlabrl/workspace/yzr/catkin_ws/src/ros1/navigation_runner/scripts/ckpts/checkpoint_0311.pt"
    # policy.load_state_dict(torch.load(checkpoint))

    # Episode Stats Collector
    episode_stats_keys = [
        k for k in transformed_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(episode_stats_keys)

    # RL Data Collector
    collector = SyncDataCollector(
        transformed_env,
        policy=policy, 
        frames_per_batch=cfg.env.num_envs * cfg.algo.training_frame_num, 
        total_frames=cfg.max_frame_num,
        device=cfg.device,
        return_same_td=True,
        exploration_type=ExplorationType.RANDOM,
    )

    # Training Loop
    for i, data in enumerate(collector):
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}

        # Train Policy
        train_loss_stats = policy.train(data)
        info.update(train_loss_stats)

        # Calculate and log training episode stats
        episode_stats.add(data)
        if len(episode_stats) >= transformed_env.num_envs:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item() 
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

        # Evaluate policy and log info
        if i % cfg.eval_interval == 0 and i > 0:
            print("[NavRL]: start evaluating policy at training step: ", i)
            env.enable_render(True)
            env.eval()
            eval_info = evaluate(
                env=transformed_env, 
                policy=policy,
                seed=cfg.seed, 
                cfg=cfg,
                exploration_type=ExplorationType.MEAN
            )
            env.enable_render(not cfg.headless)
            env.train()
            env.reset()
            info.update(eval_info)
            print("\n[NavRL]: evaluation done.")
        
        # # Save Model
        # if i % cfg.save_interval == 0:
        #     ckpt_path = os.path.join("checkpoints", f"checkpoint_{i}.pt")
        #     torch.save(policy.state_dict(), ckpt_path)
        #     print("[NavRL]: model saved at training step: ", i)

    ckpt_path = os.path.join("checkpoints", "checkpoint_final.pt")
    torch.save(policy.state_dict(), ckpt_path)
    sim_app.close()

if __name__ == "__main__":
    main()