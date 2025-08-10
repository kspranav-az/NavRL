#!/usr/bin/env python3
"""
Debug script to test hover assistance and drone behavior
"""
import torch
import argparse
from omegaconf import DictConfig, OmegaConf

try:
    from isaacsim import SimulationApp  # Isaac Sim / Isaac Lab newer namespace
except ImportError:
    from omni.isaac.kit import SimulationApp  # Fallback to legacy namespace

def test_hover_assistance():
    """Test the hover assistance mechanism"""
    
    # Create minimal simulation
    sim_app = SimulationApp({"headless": False, "anti_aliasing": 1})
    
    # Import after SimulationApp
    from env import NavigationEnv
    
    # Load config
    from hydra import initialize_config_store, compose
    
    # Create a test config
    cfg = OmegaConf.create({
        'device': 'cuda:0',
        'headless': False,
        'env': {
            'num_envs': 3,
            'max_episode_length': 1000,
            'env_spacing': 5.0,
            'num_obstacles': 0,
            'hover_assistance': True,
            'sample_spawn_area': 'rectangle',
            'sample_target_area': 'circle'
        },
        'env_dyn': {
            'num_obstacles': 0
        },
        'sensor': {
            'lidar_range': 10.0,
            'lidar_vfov': [-30, 30],
            'lidar_vbeams': 16,
            'lidar_hres': 10.0
        },
        'drone': {
            'model_name': 'Hummingbird'
        },
        'sim': {
            'dt': 0.02,
            'substeps': 1
        }
    })
    
    try:
        # Create environment
        print("[Debug] Creating environment...")
        env = NavigationEnv(cfg)
        
        print("[Debug] Environment created successfully!")
        print(f"[Debug] Number of drones: {env.num_envs}")
        print(f"[Debug] Hover assistance: {getattr(cfg.env, 'hover_assistance', 'Not specified')}")
        
        # Reset environment
        print("[Debug] Resetting environment...")
        tensordict = env.reset()
        
        print("[Debug] Environment reset successfully!")
        print(f"[Debug] Observation shape: {tensordict.shape}")
        
        # Test with different action magnitudes
        test_actions = [
            torch.zeros(env.num_envs, 1, 3, device=env.device),  # Zero actions
            torch.ones(env.num_envs, 1, 3, device=env.device) * 0.05,  # Very small actions
            torch.ones(env.num_envs, 1, 3, device=env.device) * 0.2,   # Medium actions
            torch.ones(env.num_envs, 1, 3, device=env.device) * 0.5,   # Large actions
        ]
        
        action_names = ["Zero", "Very Small", "Medium", "Large"]
        
        for i, (test_action, name) in enumerate(zip(test_actions, action_names)):
            print(f"\n[Debug] Testing {name} actions: {test_action[0, 0]}")
            
            # Create tensordict with test action
            test_tensordict = tensordict.clone()
            test_tensordict[("agents", "action")] = test_action
            
            # Get initial drone position
            initial_pos = env.drone.get_world_poses(clone=True)[0][:, 0, 2]  # Z positions
            print(f"[Debug] Initial heights: {initial_pos}")
            
            # Run a few steps
            for step in range(10):
                test_tensordict = env.step(test_tensordict)
                current_pos = env.drone.get_world_poses(clone=True)[0][:, 0, 2]
                
                if step % 5 == 0:
                    print(f"[Debug] Step {step}, heights: {current_pos}, avg: {current_pos.mean():.2f}")
            
            final_pos = env.drone.get_world_poses(clone=True)[0][:, 0, 2]
            height_change = final_pos - initial_pos
            print(f"[Debug] Final heights: {final_pos}")
            print(f"[Debug] Height change: {height_change}, avg change: {height_change.mean():.2f}")
            
            # Reset for next test
            tensordict = env.reset()
        
        print("\n[Debug] Hover assistance test completed!")
        
    except Exception as e:
        print(f"[Debug] Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("[Debug] Closing simulation...")
        sim_app.close()

if __name__ == "__main__":
    test_hover_assistance()
