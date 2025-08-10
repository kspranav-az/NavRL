#!/usr/bin/env python3
"""
Simple test script to verify hover assistance tensor logic
"""
import torch

def test_hover_assistance_logic():
    """Test the hover assistance tensor operations"""
    
    # Simulate the actions tensor from the environment
    actions = torch.randn(9, 1, 3)  # [9, 1, 3] - 9 environments, 1 agent, 3 velocity components
    
    print(f"Original actions shape: {actions.shape}")
    print(f"Sample actions: {actions[0, 0]}")
    
    # Compute action magnitude
    action_magnitude = torch.linalg.norm(actions, dim=-1, keepdim=True)
    print(f"Action magnitude shape: {action_magnitude.shape}")
    print(f"Sample magnitudes: {action_magnitude[:3, 0, 0]}")
    
    # Check which actions are small
    is_small_action = action_magnitude < 0.15
    print(f"Small action mask shape: {is_small_action.shape}")
    print(f"Small actions count: {is_small_action.sum().item()}")
    
    # Reshape to match actions for broadcasting
    if is_small_action.shape != actions.shape:
        if is_small_action.dim() == 4:  # [9, 1, 1, 1]
            is_small_action = is_small_action.squeeze(-1).squeeze(-1)  # [9, 1]
        elif is_small_action.dim() == 3:  # [9, 1, 1]
            is_small_action = is_small_action.squeeze(-1)  # [9, 1]
        
        # Now expand to match actions
        is_small_action = is_small_action.unsqueeze(-1).expand_as(actions)
    
    print(f"Final small action mask shape: {is_small_action.shape}")
    
    # Create hover assistance
    hover_assist = torch.zeros_like(actions)
    hover_assist[..., 2] = 0.3  # Upward velocity
    hover_assist[..., 0] = 0.1  # Forward velocity
    
    print(f"Hover assistance shape: {hover_assist.shape}")
    print(f"Sample hover assist: {hover_assist[0, 0]}")
    
    # Apply assistance
    assisted_actions = torch.where(
        is_small_action,
        actions + hover_assist * 0.8,
        actions
    )
    
    print(f"Assisted actions shape: {assisted_actions.shape}")
    print(f"Sample assisted: {assisted_actions[0, 0]}")
    
    # Test with different tensor shapes
    print("\n--- Testing different tensor shapes ---")
    
    # Test case 1: [9, 1, 1, 1] -> [9, 1, 3]
    test_tensor = torch.randn(9, 1, 1, 1)
    print(f"Test tensor [9,1,1,1]: {test_tensor.shape}")
    reshaped = test_tensor.squeeze(-1).squeeze(-1)
    print(f"After squeeze: {reshaped.shape}")
    expanded = reshaped.unsqueeze(-1).expand(9, 1, 3)
    print(f"After expand: {expanded.shape}")
    
    # Test case 2: [9, 1, 1] -> [9, 1, 3]
    test_tensor2 = torch.randn(9, 1, 1)
    print(f"Test tensor [9,1,1]: {test_tensor2.shape}")
    reshaped2 = test_tensor2.squeeze(-1)
    print(f"After squeeze: {reshaped2.shape}")
    expanded2 = reshaped2.unsqueeze(-1).expand(9, 1, 3)
    print(f"After expand: {expanded2.shape}")
    
    print("\n✅ All tensor operations completed successfully!")

if __name__ == "__main__":
    test_hover_assistance_logic()
