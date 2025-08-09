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
    # Add command line argument parsing for checkpoint path
    parser = argparse.ArgumentParser(description="Evaluate NavRL Policy")
    parser.add_argument("--checkpoint", type=str, default=None, 
                       help="Path to checkpoint file (optional)")
    parser.add_argument("--num_episodes", type=int, default=10,
                       help="Number of episodes to evaluate")
    parser.add_argument("--record_video", action="store_true",
                       help="Record evaluation video")
    args, unknown = parser.parse_known_args()
    
    print(f"[NavRL Evaluation] Starting evaluation with {cfg.env.num_envs} environments")
    print(f"[NavRL Evaluation] Number of evaluation episodes: {args.num_episodes}")
    print(f"[NavRL Evaluation] Video recording: {args.record_video}")
    # Simulation App
    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})

    # Initialize TensorBoard / CSV logger
    exp_name = generate_exp_name(cfg.get("logger", {}).get("exp_name", "NavRL"), "eval")
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
    controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=False)
    transforms.append(vel_transform)
    transformed_env = TransformedEnv(env, Compose(*transforms)).train()
    transformed_env.set_seed(cfg.seed)    
    # PPO Policy
    policy = PPO(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)

    # Load checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Look for latest checkpoint in logs directory
        possible_paths = [
            "logs/NavRL/checkpoint_latest.pt",
            "logs/checkpoint_latest.pt", 
            "checkpoint_latest.pt",
            "logs/NavRL/checkpoint_1000.pt",  # Common checkpoint names
            "logs/NavRL/checkpoint_final.pt"
        ]
        checkpoint_path = None
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[NavRL Evaluation] Loading checkpoint from: {checkpoint_path}")
        policy.load_state_dict(torch.load(checkpoint_path, map_location=cfg.device))
    else:
        print(f"[NavRL Evaluation] Warning: No checkpoint found!")
        print("[NavRL Evaluation] Running evaluation with randomly initialized policy")
        if args.checkpoint:
            print(f"[NavRL Evaluation] Specified checkpoint: {args.checkpoint} not found")
    
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

    # Evaluation Loop (simplified for evaluation script)
    print(f"\n[NavRL Evaluation] Starting evaluation with {args.num_episodes} episodes...")
    
    # Run evaluation episodes
    env.eval()
    
    total_episodes = 0
    all_eval_results = []
    
    for episode in range(args.num_episodes):
        print(f"\n[NavRL Evaluation] Episode {episode + 1}/{args.num_episodes}")
        
        # Reset environment
        env.reset()
        
        # Evaluate policy
        eval_info = evaluate(
            env=transformed_env, 
            policy=policy,
            seed=cfg.seed + episode,  # Different seed for each episode 
            cfg=cfg,
            exploration_type=ExplorationType.MEAN
        )
        
        all_eval_results.append(eval_info)
        
        # Print episode results
        if "eval/return" in eval_info:
            print(f"[NavRL Evaluation] Episode {episode + 1} Return: {eval_info['eval/return']:.3f}")
        if "eval/episode_len" in eval_info:
            print(f"[NavRL Evaluation] Episode {episode + 1} Length: {eval_info['eval/episode_len']:.1f}")
        
        total_episodes += 1
    
    # Calculate and log average results
    print(f"\n[NavRL Evaluation] ===== FINAL RESULTS =====")
    if all_eval_results:
        avg_results = {}
        for key in all_eval_results[0].keys():
            if key != "recording":  # Skip video recording for averaging
                values = [result[key] for result in all_eval_results if key in result]
                if values and isinstance(values[0], (int, float, torch.Tensor)):
                    avg_value = sum(values) / len(values)
                    if isinstance(avg_value, torch.Tensor):
                        avg_value = avg_value.item()
                    avg_results[f"avg_{key}"] = avg_value
                    print(f"[NavRL Evaluation] Average {key}: {avg_value:.3f}")
        
        # Log to TensorBoard
        for k, v in avg_results.items():
            logger.log_scalar(k, float(v), step=0)
        
        # Handle video recording if enabled
        if args.record_video and "recording" in all_eval_results[-1]:
            try:
                import numpy as _np
                import torch as _torch
                vid = all_eval_results[-1]["recording"]
                if isinstance(vid, _np.ndarray):
                    vid = _torch.from_numpy(vid)
                if vid.ndim == 4:
                    vid = vid.unsqueeze(0)
                fps = int(0.5 / (cfg.sim.dt * cfg.sim.substeps))
                logger.log_video("eval/recording", vid, step=0, fps=fps)
                print("[NavRL Evaluation] Video recorded and logged to TensorBoard")
            except Exception as e:
                print(f"[NavRL Evaluation] Failed to record video: {e}")
    
    print(f"\n[NavRL Evaluation] Evaluation completed successfully!")
    print(f"[NavRL Evaluation] Total episodes evaluated: {total_episodes}")
    print(f"[NavRL Evaluation] Results logged to TensorBoard in: {logger.experiment_name}")
    
    env.close()

    sim_app.close()

if __name__ == "__main__":
    main()
    