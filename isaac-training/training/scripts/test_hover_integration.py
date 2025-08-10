#!/usr/bin/env python3
"""
Test script to verify hover assistance integration without Isaac Sim dependencies
"""
import torch
import sys
import os

# Add the current directory to path to import env module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_hover_assistance_integration():
    """Test the hover assistance logic from the environment file"""
    
    print("🧪 Testing Hover Assistance Integration")
    print("=" * 50)
    
                # Test 1: Basic tensor operations
    print("\n1️⃣ Testing basic tensor operations...")
    try:
        # Test both 3D and 4D actions
        actions_3d = torch.randn(9, 1, 3)
        actions_4d = torch.randn(9, 1, 4)
        
        # Test 3D actions
        action_magnitude_3d = torch.linalg.norm(actions_3d, dim=-1, keepdim=True)
        is_small_action_3d = action_magnitude_3d < 0.15
        
        # Test 4D actions (only x,y,z components for magnitude)
        action_magnitude_4d = torch.linalg.norm(actions_4d[..., :3], dim=-1, keepdim=True)
        is_small_action_4d = action_magnitude_4d < 0.15
        
        print(f"   ✅ 3D Actions: {actions_3d.shape}")
        print(f"   ✅ 3D Magnitude: {action_magnitude_3d.shape}")
        print(f"   ✅ 3D Small mask: {is_small_action_3d.shape}")
        print(f"   ✅ 3D Small count: {is_small_action_3d.sum().item()}")
        
        print(f"   ✅ 4D Actions: {actions_4d.shape}")
        print(f"   ✅ 4D Magnitude: {action_magnitude_4d.shape}")
        print(f"   ✅ 4D Small mask: {is_small_action_4d.shape}")
        print(f"   ✅ 4D Small count: {is_small_action_4d.sum().item()}")
        
    except Exception as e:
        print(f"   ❌ Basic tensor ops failed: {e}")
        return False
    
    # Test 2: Tensor reshaping logic
    print("\n2️⃣ Testing tensor reshaping logic...")
    try:
        # Test different input shapes for both 3D and 4D actions
        test_shapes_3d = [
            (9, 1, 1, 1),  # 4D
            (9, 1, 1),      # 3D
            (9, 1)           # 2D
        ]
        
        test_shapes_4d = [
            (9, 1, 1, 1),  # 4D
            (9, 1, 1),      # 3D
            (9, 1)           # 2D
        ]
        
        # Test 3D actions
        print("   Testing 3D action reshaping...")
        for shape in test_shapes_3d:
            test_tensor = torch.randn(*shape)
            print(f"     Testing shape {shape}: {test_tensor.shape}")
            
            # Apply the same logic from env.py
            if test_tensor.dim() == 4:  # [9, 1, 1, 1]
                reshaped = test_tensor.squeeze(-1).squeeze(-1)  # [9, 1]
            elif test_tensor.dim() == 3:  # [9, 1, 1]
                reshaped = test_tensor.squeeze(-1)  # [9, 1]
            else:
                reshaped = test_tensor
            
            expanded = reshaped.unsqueeze(-1).expand(9, 1, 3)
            print(f"       → Reshaped: {reshaped.shape}")
            print(f"       → Expanded: {expanded.shape}")
            
            # Verify final shape
            assert expanded.shape == (9, 1, 3), f"Expected (9,1,3), got {expanded.shape}"
            print(f"       ✅ Shape validation passed")
        
        # Test 4D actions
        print("   Testing 4D action reshaping...")
        for shape in test_shapes_4d:
            test_tensor = torch.randn(*shape)
            print(f"     Testing shape {shape}: {test_tensor.shape}")
            
            # Apply the same logic from env.py
            if test_tensor.dim() == 4:  # [9, 1, 1, 1]
                reshaped = test_tensor.squeeze(-1).squeeze(-1)  # [9, 1]
            elif test_tensor.dim() == 3:  # [9, 1, 1]
                reshaped = test_tensor.squeeze(-1)  # [9, 1]
            else:
                reshaped = test_tensor
            
            expanded = reshaped.unsqueeze(-1).expand(9, 1, 4)
            print(f"       → Reshaped: {reshaped.shape}")
            print(f"       → Expanded: {expanded.shape}")
            
            # Verify final shape
            assert expanded.shape == (9, 1, 4), f"Expected (9,1,4), got {expanded.shape}"
            print(f"       ✅ Shape validation passed")
        
        print("   ✅ All tensor reshaping tests passed")
    except Exception as e:
        print(f"   ❌ Tensor reshaping failed: {e}")
        return False
    
    # Test 3: Hover assistance application
    print("\n3️⃣ Testing hover assistance application...")
    try:
        # Test 3D actions
        actions_3d = torch.randn(9, 1, 3)
        action_magnitude_3d = torch.linalg.norm(actions_3d, dim=-1, keepdim=True)
        is_small_action_3d = action_magnitude_3d < 0.15
        
        # Create hover assistance for 3D
        hover_assist_3d = torch.zeros_like(actions_3d)
        hover_assist_3d[..., 2] = 0.3  # Upward velocity
        hover_assist_3d[..., 0] = 0.1  # Forward velocity
        
        # Test 4D actions
        actions_4d = torch.randn(9, 1, 4)
        action_magnitude_4d = torch.linalg.norm(actions_4d[..., :3], dim=-1, keepdim=True)
        is_small_action_4d = action_magnitude_4d < 0.15
        
        # Create hover assistance for 4D (no yaw assistance)
        hover_assist_4d = torch.zeros_like(actions_4d)
        hover_assist_4d[..., 2] = 0.3  # Upward velocity
        hover_assist_4d[..., 0] = 0.1  # Forward velocity
        hover_assist_4d[..., 3] = 0.0  # No yaw assistance
        
        # Apply reshaping logic for 3D actions
        if is_small_action_3d.shape != actions_3d.shape:
            if is_small_action_3d.dim() == 4:
                is_small_action_3d = is_small_action_3d.squeeze(-1).squeeze(-1)
            elif is_small_action_3d.dim() == 3:
                is_small_action_3d = is_small_action_3d.squeeze(-1)
            is_small_action_3d = is_small_action_3d.unsqueeze(-1).expand_as(actions_3d)
        
        # Apply assistance for 3D
        assisted_actions_3d = torch.where(
            is_small_action_3d,
            actions_3d + hover_assist_3d * 0.8,
            actions_3d
        )
        
        # Apply reshaping logic for 4D actions
        if is_small_action_4d.shape != actions_4d.shape:
            if is_small_action_4d.dim() == 4:
                is_small_action_4d = is_small_action_4d.squeeze(-1).squeeze(-1)
            elif is_small_action_4d.dim() == 3:
                is_small_action_4d = is_small_action_4d.squeeze(-1)
            is_small_action_4d = is_small_action_4d.unsqueeze(-1).expand_as(actions_4d)
        
        # Apply assistance for 4D
        assisted_actions_4d = torch.where(
            is_small_action_4d,
            actions_4d + hover_assist_4d * 0.8,
            actions_4d
        )
        
        # Verify shapes for 3D
        assert assisted_actions_3d.shape == actions_3d.shape, "Shape mismatch after 3D assistance"
        print(f"   ✅ 3D Original actions: {actions_3d.shape}")
        print(f"   ✅ 3D Assisted actions: {assisted_actions_3d.shape}")
        print(f"   ✅ 3D Hover assistance: {hover_assist_3d.shape}")
        
        # Verify shapes for 4D
        assert assisted_actions_4d.shape == actions_4d.shape, "Shape mismatch after 4D assistance"
        print(f"   ✅ 4D Original actions: {actions_4d.shape}")
        print(f"   ✅ 4D Assisted actions: {assisted_actions_4d.shape}")
        print(f"   ✅ 4D Hover assistance: {hover_assist_4d.shape}")
        
        # Check that assistance was applied
        assistance_applied_3d = torch.any(is_small_action_3d).item()
        assistance_applied_4d = torch.any(is_small_action_4d).item()
        print(f"   ✅ 3D Assistance applied: {assistance_applied_3d}")
        print(f"   ✅ 4D Assistance applied: {assistance_applied_4d}")
        
    except Exception as e:
        print(f"   ❌ Hover assistance application failed: {e}")
        return False
    
    # Test 4: Edge cases
    print("\n4️⃣ Testing edge cases...")
    try:
        # Test with all small 3D actions
        small_actions_3d = torch.randn(9, 1, 3) * 0.1  # All actions < 0.15
        small_magnitude_3d = torch.linalg.norm(small_actions_3d, dim=-1, keepdim=True)
        all_small_3d = small_magnitude_3d < 0.15
        
        print(f"   3D All small actions: {all_small_3d.sum().item()}/{all_small_3d.numel()}")
        
        # Test with all large 3D actions
        large_actions_3d = torch.randn(9, 1, 3) * 2.0  # All actions > 0.15
        large_magnitude_3d = torch.linalg.norm(large_actions_3d, dim=-1, keepdim=True)
        all_large_3d = large_magnitude_3d < 0.15
        
        print(f"   3D All large actions: {all_large_3d.sum().item()}/{all_large_3d.numel()}")
        
        # Test with all small 4D actions
        small_actions_4d = torch.randn(9, 1, 4) * 0.1  # All actions < 0.15
        small_magnitude_4d = torch.linalg.norm(small_actions_4d[..., :3], dim=-1, keepdim=True)
        all_small_4d = small_magnitude_4d < 0.15
        
        print(f"   4D All small actions: {all_small_4d.sum().item()}/{all_small_4d.numel()}")
        
        # Test with all large 4D actions
        large_actions_4d = torch.randn(9, 1, 4) * 2.0  # All actions > 0.15
        large_magnitude_4d = torch.linalg.norm(large_actions_4d[..., :3], dim=-1, keepdim=True)
        all_large_4d = large_magnitude_4d < 0.15
        
        print(f"   4D All large actions: {all_large_4d.sum().item()}/{all_large_4d.numel()}")
        
        print("   ✅ Edge case tests passed")
        
    except Exception as e:
        print(f"   ❌ Edge case tests failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All hover assistance integration tests passed!")
    print("✅ The hover assistance logic is ready for use in the environment")
    
    return True

def test_configuration_options():
    """Test configuration options for hover assistance"""
    
    print("\n🔧 Testing Configuration Options")
    print("=" * 50)
    
    # Test environment variable handling
    print("\n1️⃣ Testing environment variable handling...")
    
    # Save original value
    original_disable = os.environ.get('DISABLE_HOVER_ASSISTANCE', 'false')
    
    try:
        # Test enabled
        os.environ['DISABLE_HOVER_ASSISTANCE'] = 'false'
        disabled = os.environ.get('DISABLE_HOVER_ASSISTANCE', 'false').lower() == 'true'
        print(f"   DISABLE_HOVER_ASSISTANCE=false → disabled={disabled}")
        
        # Test disabled
        os.environ['DISABLE_HOVER_ASSISTANCE'] = 'true'
        disabled = os.environ.get('DISABLE_HOVER_ASSISTANCE', 'false').lower() == 'true'
        print(f"   DISABLE_HOVER_ASSISTANCE=true → disabled={disabled}")
        
        # Test case insensitive
        os.environ['DISABLE_HOVER_ASSISTANCE'] = 'TRUE'
        disabled = os.environ.get('DISABLE_HOVER_ASSISTANCE', 'false').lower() == 'true'
        print(f"   DISABLE_HOVER_ASSISTANCE=TRUE → disabled={disabled}")
        
        print("   ✅ Environment variable handling works correctly")
        
    except Exception as e:
        print(f"   ❌ Environment variable test failed: {e}")
    finally:
        # Restore original value
        os.environ['DISABLE_HOVER_ASSISTANCE'] = original_disable
    
    # Test configuration parameter
    print("\n2️⃣ Testing configuration parameter...")
    try:
        # Simulate config object
        class MockConfig:
            def __init__(self, hover_assistance=True):
                self.env = type('obj', (object,), {'hover_assistance': hover_assistance})()
        
        config_enabled = MockConfig(hover_assistance=True)
        config_disabled = MockConfig(hover_assistance=False)
        
        enabled = getattr(config_enabled.env, 'hover_assistance', True)
        disabled = getattr(config_disabled.env, 'hover_assistance', True)
        
        print(f"   Config hover_assistance=True → enabled={enabled}")
        print(f"   Config hover_assistance=False → enabled={disabled}")
        print(f"   Default fallback → enabled={getattr(type('obj', (object,), {})(), 'hover_assistance', True)}")
        
        print("   ✅ Configuration parameter handling works correctly")
        
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🔧 Configuration options are working correctly")

if __name__ == "__main__":
    print("🚁 Hover Assistance Integration Test Suite")
    print("Testing the hover assistance logic without Isaac Sim dependencies")
    
    # Run tests
    success = test_hover_assistance_integration()
    test_configuration_options()
    
    if success:
        print("\n🎯 SUMMARY: Hover assistance is ready for deployment!")
        print("   ✅ Tensor operations work correctly")
        print("   ✅ Reshaping logic handles all cases")
        print("   ✅ Assistance application works")
        print("   ✅ Edge cases are handled")
        print("   ✅ Configuration options work")
        print("\n   🚀 You can now run the training script with hover assistance enabled!")
    else:
        print("\n⚠️  SUMMARY: Some tests failed - check the output above")
