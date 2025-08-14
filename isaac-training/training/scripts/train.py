import argparse
import os
import hydra
import datetime
import torch
from omegaconf import DictConfig, OmegaConf
try:
    from isaacsim import SimulationApp  # Isaac Sim / Isaac Lab newer namespace
except ImportError:
    from omni.isaac.kit import SimulationApp  # Fallback to legacy namespace
from ppo import PPO
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torchrl.transforms import VelController, ravel_composite
from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
from torchrl.envs.transforms import TransformedEnv, Compose
from utils import evaluate
from torchrl.envs.utils import ExplorationType
from torchrl.record.loggers import get_logger, generate_exp_name




FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")
@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    # Simulation App
    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})

    # Initialize TensorBoard / CSV logger
    exp_name = generate_exp_name(cfg.get("logger", {}).get("exp_name", "NavRL"), "train")
    logger = get_logger(
        logger_type=cfg.get("logger", {}).get("backend", "tensorboard"),
        logger_name=cfg.get("logger", {}).get("log_dir", "logs"),
        experiment_name=exp_name,
    )
    try:
        logger.log_hparams(cfg)
    except Exception:
        pass

    # Navigation Training Environment
    from env import NavigationEnv
    env = NavigationEnv(cfg)

    # Transformed Environment
    transforms = []
    # transforms.append(ravel_composite(env.observation_spec, ("agents", "intrinsics"), start_dim=-1))
    
    # Add hover assistance transform before the controller
    from torchrl.envs.transforms import Transform
    class HoverAssistanceTransform(Transform):
        def __init__(self, action_key: str = ("agents", "action")):
            super().__init__([], in_keys_inv=[])
            self.action_key = action_key
        
        def _inv_call(self, tensordict) -> "TensorDict":
            actions = tensordict[self.action_key]
            
            # Apply hover assistance to 3D velocity actions
            if actions.shape[-1] == 3:  # 3D velocity actions
                # Compute action magnitude
                action_magnitude = torch.linalg.norm(actions, dim=-1, keepdim=True)
                is_small_action = action_magnitude < 0.35
                
                # Create hover assistance (strong upward velocity)
                hover_assist = torch.zeros_like(actions)
                hover_assist[..., 2] = 0.8  # Strong upward velocity
                hover_assist[..., 0] = 0.02  # Minimal forward velocity
                
                # Apply assistance where actions are small
                assisted_actions = torch.where(
                    is_small_action,
                    actions + hover_assist * 0.8,
                    actions
                )
                
                # Ensure minimum action magnitude
                final_magnitude = torch.linalg.norm(assisted_actions, dim=-1, keepdim=True)
                min_safe_magnitude = 0.05
                safe_actions = torch.where(
                    final_magnitude < min_safe_magnitude,
                    assisted_actions * (min_safe_magnitude / (final_magnitude + 1e-8)),
                    assisted_actions
                )
                
                tensordict.set(self.action_key, safe_actions)
            
            return tensordict
    
    # Add transforms in order: hover assistance first, then controller
    transforms.append(HoverAssistanceTransform())
    controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=False)
    transforms.append(vel_transform)
    transformed_env = TransformedEnv(env, Compose(*transforms)).train()
    transformed_env.set_seed(cfg.seed)    
    # PPO Policy
    policy = PPO(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)

    # checkpoint = "/home/zhefan/catkin_ws/src/navigation_runner/scripts/ckpts/checkpoint_2500.pt"
    # checkpoint = "/home/xinmingh/RLDrones/navigation/scripts/nav-ros/navigation_runner/ckpts/checkpoint_36000.pt"
    # policy.load_state_dict(torch.load(checkpoint))
    
    # Episode Stats Collector
    episode_stats_keys = [
        k for k in transformed_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(episode_stats_keys)

    # RL Data Collector
    collector = SyncDataCollector(
        transformed_env,
        policy=policy, 
        frames_per_batch=cfg.env.num_envs * cfg.algo.training_frame_num, 
        total_frames=cfg.max_frame_num,
        device=cfg.device,
        return_same_td=True, # update the return tensordict inplace (should set to false if we need to use replace buffer)
        exploration_type=ExplorationType.RANDOM, # sample from normal distribution
    )

    # Training Loop
    for i, data in enumerate(collector):
        # print("data: ", data)
        # print("============================")
        # Log Info
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}

        # Train Policy
        train_loss_stats = policy.train(data)
        info.update(train_loss_stats) # log training loss info

        # Calculate and log training episode stats
        episode_stats.add(data)
        if len(episode_stats) >= transformed_env.num_envs: # evaluate once if all agents finished one episode
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item() 
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

        # Evaluate policy and log info
        if i % cfg.eval_interval == 0:
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
        
        # Log info via logger: scalars and optional video
        # Scalars
        for k, v in list(info.items()):
            if k == "recording":
                continue
            if isinstance(v, torch.Tensor):
                if v.ndim == 0:
                    v = v.item()
                else:
                    continue
            if isinstance(v, (int, float)):
                logger.log_scalar(k, float(v), step=i)

        # Video
        if "recording" in info:
            try:
                import numpy as _np
                import torch as _torch
                vid = info["recording"]
                if isinstance(vid, _np.ndarray):
                    vid = _torch.from_numpy(vid)
                if vid.ndim == 4:  # T, C, H, W -> add batch dim
                    vid = vid.unsqueeze(0)
                fps = int(0.5 / (cfg.sim.dt * cfg.sim.substeps))
                logger.log_video("eval/recording", vid, step=i, fps=fps)
            except Exception:
                pass


        # Save Model
        if i % cfg.save_interval == 0:
            ckpt_dir = os.path.join(logger.log_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{i}.pt")
            torch.save(policy.state_dict(), ckpt_path)
            print("[NavRL]: model saved at training step: ", i)

    ckpt_dir = os.path.join(logger.log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "checkpoint_final.pt")
    torch.save(policy.state_dict(), ckpt_path)
    sim_app.close()

if __name__ == "__main__":
    main()
    