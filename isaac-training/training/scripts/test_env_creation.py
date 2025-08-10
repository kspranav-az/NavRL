#!/usr/bin/env python3
"""
Test script to verify environment creation works
"""
import torch
from omegaconf import OmegaConf

def test_env_creation():
    """Test if the NavigationEnv can be created without errors"""
    
    # Create a minimal config
    cfg = OmegaConf.create({
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'headless': True,  # Headless for testing
        'env': {
            'num_envs': 3,  # Small number for testing
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
    
    print("✅ Config created successfully")
    print(f"Device: {cfg.device}")
    print(f"Number of environments: {cfg.env.num_envs}")
    print(f"Hover assistance: {cfg.env.hover_assistance}")
    
    try:
        # Try to import the environment
        print("\n🔄 Importing environment...")
        from env import NavigationEnv
        print("✅ Environment imported successfully")
        
        # Try to create the environment
        print("\n🔄 Creating environment...")
        env = NavigationEnv(cfg)
        print("✅ Environment created successfully!")
        
        # Try to reset the environment
        print("\n🔄 Resetting environment...")
        tensordict = env.reset()
        print("✅ Environment reset successfully!")
        print(f"Observation shape: {tensordict.shape}")
        
        # Try to get a step
        print("\n🔄 Testing environment step...")
        test_action = torch.zeros(env.num_envs, 1, 3, device=env.device)
        test_tensordict = tensordict.clone()
        test_tensordict[("agents", "action")] = test_action
        
        result = env.step(test_tensordict)
        print("✅ Environment step completed successfully!")
        
        print("\n🎉 All tests passed! Environment is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_env_creation()
    if success:
        print("\n🚀 Environment is ready for training!")
    else:
        print("\n⚠️  Environment has issues that need to be fixed.")
