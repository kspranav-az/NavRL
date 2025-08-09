# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
from typing import Optional, Tuple, List
from contextlib import contextmanager

from typing import List, Optional, Tuple, Union
import numpy as np
import carb
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
try:
    from isaacsim.utils.prims import get_prim_parent, get_prim_at_path, set_prim_property, get_prim_property
    from isaacsim.utils.types import JointsState, ArticulationActions
    from isaacsim.articulations import ArticulationView as _ArticulationView
    from isaacsim.prims import RigidPrimView as _RigidPrimView
    from isaacsim.prims import XFormPrimView
    from isaacsim.simulation_context import SimulationContext
except ImportError:
    from omni.isaac.core.utils.prims import get_prim_parent, get_prim_at_path, set_prim_property, get_prim_property
    from omni.isaac.core.utils.types import JointsState, ArticulationActions
    from omni.isaac.core.articulations import ArticulationView as _ArticulationView
    from omni.isaac.core.prims import RigidPrimView as _RigidPrimView
    from omni.isaac.core.prims import XFormPrimView
    from omni.isaac.core.simulation_context import SimulationContext
import omni
import functools

print("[OmniDrones] views/__init__.py is being loaded - patches will be applied")

# Monkey patch ArticulationView to fix get_world_poses API compatibility issue
# This addresses the "TypeError: ArticulationView.get_world_poses() got an unexpected keyword argument 'usd'"
try:
    # Get the actual ArticulationView class that was imported
    _original_articulation_view_get_world_poses = _ArticulationView.get_world_poses
    def _patched_get_world_poses(self, *args, **kwargs):
        # Remove 'usd' parameter if present to maintain compatibility
        kwargs.pop('usd', None)
        return _original_articulation_view_get_world_poses(self, *args, **kwargs)
    _ArticulationView.get_world_poses = _patched_get_world_poses
    print(f"[OmniDrones] Applied ArticulationView.get_world_poses compatibility patch to {_ArticulationView.__module__}.{_ArticulationView.__name__}")
except Exception as e:
    print(f"[OmniDrones] Failed to apply ArticulationView patch: {e}")
    
# Additional patch: Also patch the base isaacsim ArticulationView if it exists and is different
try:
    from isaacsim.core.prims import ArticulationView as IsaacSimArticulationView
    if IsaacSimArticulationView != _ArticulationView:
        _original_isaacsim_get_world_poses = IsaacSimArticulationView.get_world_poses
        def _patched_isaacsim_get_world_poses(self, *args, **kwargs):
            kwargs.pop('usd', None)  # Remove 'usd' parameter if present
            return _original_isaacsim_get_world_poses(self, *args, **kwargs)
        IsaacSimArticulationView.get_world_poses = _patched_isaacsim_get_world_poses
        print(f"[OmniDrones] Applied additional patch to {IsaacSimArticulationView.__module__}.{IsaacSimArticulationView.__name__}")
except ImportError:
    print("[OmniDrones] isaacsim.core.prims.ArticulationView not available for patching")
except Exception as e:
    print(f"[OmniDrones] Failed to apply additional ArticulationView patch: {e}")

# Comprehensive patch: Try to patch XFormPrimView and any other related classes
try:
    # Patch XFormPrimView if it exists
    if hasattr(XFormPrimView, 'get_world_poses'):
        _original_xform_get_world_poses = XFormPrimView.get_world_poses
        def _patched_xform_get_world_poses(self, *args, **kwargs):
            kwargs.pop('usd', None)  # Remove 'usd' parameter if present
            return _original_xform_get_world_poses(self, *args, **kwargs)
        XFormPrimView.get_world_poses = _patched_xform_get_world_poses
        print(f"[OmniDrones] Applied patch to {XFormPrimView.__module__}.{XFormPrimView.__name__}")
except Exception as e:
    print(f"[OmniDrones] Failed to patch XFormPrimView: {e}")

# Also try to patch the XFormPrim class from isaacsim.core
try:
    from isaacsim.core.prims.impl.xform_prim import XFormPrim
    if hasattr(XFormPrim, 'get_world_poses'):
        _original_xform_prim_get_world_poses = XFormPrim.get_world_poses
        def _patched_xform_prim_get_world_poses(self, *args, **kwargs):
            kwargs.pop('usd', None)  # Remove 'usd' parameter if present
            return _original_xform_prim_get_world_poses(self, *args, **kwargs)
        XFormPrim.get_world_poses = _patched_xform_prim_get_world_poses
        print(f"[OmniDrones] Applied patch to {XFormPrim.__module__}.{XFormPrim.__name__}")
except ImportError:
    print("[OmniDrones] isaacsim.core.prims.impl.xform_prim.XFormPrim not available for patching")
except Exception as e:
    print(f"[OmniDrones] Failed to patch XFormPrim: {e}")


def require_sim_initialized(func):

    @functools.wraps(func)
    def _func(*args, **kwargs):
        # Check if simulation context is ready with proper fallback
        try:
            from isaacsim.core.simulation_manager import SimulationManager
            physics_sim_view = SimulationManager.get_physics_sim_view()
            if physics_sim_view is None:
                raise RuntimeError("SimulationContext not initialized.")
        except ImportError:
            # Fallback for older versions
            if not hasattr(SimulationContext.instance(), '_physics_sim_view') or SimulationContext.instance()._physics_sim_view is None:
                raise RuntimeError("SimulationContext not initialized.")
        return func(*args, **kwargs)
    
    return _func


class ArticulationView(_ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: str = "articulation_prim_view",
        positions: Optional[torch.Tensor] = None,
        translations: Optional[torch.Tensor] = None,
        orientations: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
        visibilities: Optional[torch.Tensor] = None,
        reset_xform_properties: bool = True,
        enable_dof_force_sensors: bool = False,
        shape: Tuple[int, ...] = (-1,),
    ) -> None:
        print(f"[OmniDrones] ArticulationView.__init__ called with prim_paths_expr: {prim_paths_expr}")
        self.shape = shape
        
        # Store enable_dof_force_sensors parameter for later use in initialize()
        self._enable_dof_force_sensors = enable_dof_force_sensors
        
        # Initialize _physics_sim_view to None to prevent AttributeError during parent initialization
        self._physics_sim_view = None
        
        # Define the get_world_poses override before calling super().__init__
        # This ensures it's available when the parent class calls it during initialization
        def get_world_poses_override(*args, **kwargs):
            """Override get_world_poses to handle API compatibility issues."""
            if 'usd' in kwargs:
                print(f"[OmniDrones] Removing 'usd' parameter from get_world_poses call")
                kwargs.pop('usd', None)  # Remove 'usd' parameter if present
            return _ArticulationView.get_world_poses(self, *args, **kwargs)
        
        # Temporarily set our override method
        original_method = getattr(self, 'get_world_poses', None)
        self.get_world_poses = get_world_poses_override
        
        try:
            super().__init__(
                prim_paths_expr,
                name,
                positions,
                translations,
                orientations,
                scales,
                visibilities,
                reset_xform_properties,
                enable_dof_force_sensors,
            )
        finally:
            # Restore the original method or keep our override
            if original_method:
                self.get_world_poses = original_method
            # If no original method, our override will remain
    
    def get_world_poses(self, *args, **kwargs):
        """Override get_world_poses to handle API compatibility issues.
        
        This method removes the 'usd' parameter that is no longer supported in newer Isaac Sim versions.
        """
        # Remove 'usd' parameter if present to maintain compatibility
        kwargs.pop('usd', None)
        return super().get_world_poses(*args, **kwargs)
    
    @require_sim_initialized
    def initialize(self, physics_sim_view: omni.physics.tensors.SimulationView = None) -> None:
        """Create a physics simulation view if not passed and creates an articulation view using physX tensor api.

        Args:
            physics_sim_view (omni.physics.tensors.SimulationView, optional): current physics simulation view. Defaults to None.
        """
        if physics_sim_view is None:
            physics_sim_view = omni.physics.tensors.create_simulation_view(self._backend)
            physics_sim_view.set_subspace_roots("/")
        carb.log_info("initializing view for {}".format(self._name))
        # Debug: Print available attributes to understand the class structure
        print(f"[OmniDrones] ArticulationView initialize - available attributes: {[attr for attr in dir(self) if '_prim' in attr.lower() or '_path' in attr.lower()]}")
        
        # Debug: Check the actual values of key attributes
        if hasattr(self, '_regex_prim_paths'):
            print(f"[OmniDrones] _regex_prim_paths exists: {self._regex_prim_paths} (type: {type(self._regex_prim_paths)})")
        if hasattr(self, '_prim_paths'):
            print(f"[OmniDrones] _prim_paths exists: {self._prim_paths} (type: {type(self._prim_paths)})")
        
        # TODO: add a callback to set physics view to None once stop is called
        # Fix: Use _regex_prim_paths (confirmed to exist as string) and convert to physics pattern
        prim_path_pattern = None
        
        # Priority 1: Check if _regex_prim_paths is a string (the pattern we want)
        if hasattr(self, '_regex_prim_paths') and isinstance(self._regex_prim_paths, str):
            prim_path_pattern = self._regex_prim_paths.replace(".*", "*")
            print(f"[OmniDrones] Using _regex_prim_paths: {self._regex_prim_paths} -> {prim_path_pattern}")
        
        # Priority 2: If _regex_prim_paths is a list, try to extract pattern from it
        elif hasattr(self, '_regex_prim_paths') and isinstance(self._regex_prim_paths, list) and len(self._regex_prim_paths) > 0:
            # Try to reconstruct the pattern from the first path
            first_path = self._regex_prim_paths[0]
            if "/env_" in first_path and "/Hummingbird_" in first_path:
                # Convert specific path back to pattern: /World/envs/env_0/Hummingbird_0 -> /World/envs/*/Hummingbird*
                pattern_parts = first_path.split("/")
                for i, part in enumerate(pattern_parts):
                    if part.startswith("env_"):
                        pattern_parts[i] = "*"
                    elif part.startswith("Hummingbird_"):
                        pattern_parts[i] = "Hummingbird*"
                prim_path_pattern = "/".join(pattern_parts)
                print(f"[OmniDrones] Reconstructed pattern from _regex_prim_paths[0]: {first_path} -> {prim_path_pattern}")
            else:
                prim_path_pattern = first_path.replace(".*", "*")
                print(f"[OmniDrones] Using _regex_prim_paths[0]: {first_path} -> {prim_path_pattern}")
        
        # Priority 3: Fallback to _prim_paths if available
        elif hasattr(self, '_prim_paths') and isinstance(self._prim_paths, list) and len(self._prim_paths) > 0:
            first_path = self._prim_paths[0]
            if "/env_" in first_path and "/Hummingbird_" in first_path:
                # Convert specific path back to pattern
                pattern_parts = first_path.split("/")
                for i, part in enumerate(pattern_parts):
                    if part.startswith("env_"):
                        pattern_parts[i] = "*"
                    elif part.startswith("Hummingbird_"):
                        pattern_parts[i] = "Hummingbird*"
                prim_path_pattern = "/".join(pattern_parts)
                print(f"[OmniDrones] Reconstructed pattern from _prim_paths[0]: {first_path} -> {prim_path_pattern}")
            else:
                prim_path_pattern = first_path.replace(".*", "*")
                print(f"[OmniDrones] Using _prim_paths[0]: {first_path} -> {prim_path_pattern}")
        
        # Priority 4: Last resort fallback
        else:
            prim_path_pattern = "/World/envs/*/Hummingbird*"
            print(f"[OmniDrones] Using fallback pattern: {prim_path_pattern}")
        
        # Ensure the pattern is valid for multiple environment instances
        if prim_path_pattern and not ("*" in prim_path_pattern or ".*" in prim_path_pattern):
            print(f"[OmniDrones] WARNING: Pattern {prim_path_pattern} doesn't contain wildcards, this may cause issues with multiple environments")
            # Force wildcard pattern
            if "/env_" in prim_path_pattern and "/Hummingbird_" in prim_path_pattern:
                pattern_parts = prim_path_pattern.split("/")
                for i, part in enumerate(pattern_parts):
                    if part.startswith("env_"):
                        pattern_parts[i] = "*"
                    elif part.startswith("Hummingbird_"):
                        pattern_parts[i] = "Hummingbird*"
                prim_path_pattern = "/".join(pattern_parts)
                print(f"[OmniDrones] Fixed pattern to support multiple environments: {prim_path_pattern}")
        
        # Create ArticulationView with just the path pattern (following IsaacLab style)
        # Note: IsaacLab's create_articulation_view only takes the path pattern, not enable_dof_force_sensors
        print(f"[OmniDrones] Creating ArticulationView with pattern: {prim_path_pattern}")
        try:
            # Try with enable_dof_force_sensors parameter first
            self._physics_view = physics_sim_view.create_articulation_view(
                prim_path_pattern, self._enable_dof_force_sensors
            )
        except TypeError as e:
            # If the signature doesn't support enable_dof_force_sensors, try without it
            print(f"[OmniDrones] create_articulation_view doesn't support enable_dof_force_sensors: {e}")
            print(f"[OmniDrones] Retrying without enable_dof_force_sensors parameter")
            self._physics_view = physics_sim_view.create_articulation_view(prim_path_pattern)
        
        print(f"[OmniDrones] ArticulationView created successfully")
        assert self._physics_view.is_homogeneous
        self._physics_sim_view = physics_sim_view
        if not self._is_initialized:
            self._metadata = self._physics_view.shared_metatype
            self._num_dof = self._physics_view.max_dofs
            self._num_bodies = self._physics_view.max_links
            self._num_shapes = self._physics_view.max_shapes
            self._num_fixed_tendons = self._physics_view.max_fixed_tendons
            self._body_names = self._metadata.link_names
            self._body_indices = dict(zip(self._body_names, range(len(self._body_names))))
            self._dof_names = self._metadata.dof_names
            self._dof_indices = self._metadata.dof_indices
            self._dof_types = self._metadata.dof_types
            self._dof_paths = self._physics_view.dof_paths
            self._prim_paths = self._physics_view.prim_paths
            carb.log_info("Articulation Prim View Device: {}".format(self._device))
            self._is_initialized = True
            self._default_kps, self._default_kds = self.get_gains(clone=True)
            default_actions = self.get_applied_actions(clone=True)
            # TODO: implement effort part
            if self.num_dof > 0:
                if self._default_joints_state is None:
                    self._default_joints_state = JointsState(positions=None, velocities=None, efforts=None)
                if self._default_joints_state.positions is None:
                    self._default_joints_state.positions = default_actions.joint_positions
                if self._default_joints_state.velocities is None:
                    self._default_joints_state.velocities = default_actions.joint_velocities
                if self._default_joints_state.efforts is None:
                    self._default_joints_state.efforts = self._backend_utils.create_zeros_tensor(
                        shape=[self.count, self.num_dof], dtype="float32", device=self._device
                    )
        return
    
    def get_gains(
        self,
        indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
        joint_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
        clone: bool = True,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Gets stiffness and damping of articulations in the view.

        Args:
            indices (Optional[Union[np.ndarray, List, torch.Tensor]], optional): indicies to specify which prims 
                                                                                 to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            joint_indices (Optional[Union[np.ndarray, List, torch.Tensor]], optional): joint indicies to specify which joints 
                                                                                 to query. Shape (K,).
                                                                                 Where K <= num of dofs.
                                                                                 Defaults to None (i.e: all dofs).
            clone (bool, optional): True to return clones of the internal buffers. Otherwise False. Defaults to True.

        Returns:
            Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]: stiffness and damping of
                                                             articulations in the view respectively. shapes are (M, K).
        """
        if not self._is_initialized:
            carb.log_warn("ArticulationView needs to be initialized.")
            return None
        if not omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
            indices = self._backend_utils.resolve_indices(indices, self.count, device="cpu")
            joint_indices = self._backend_utils.resolve_indices(joint_indices, self.num_dof, device="cpu")
            if joint_indices.numel() == 0:
                return None, None
            kps = self._physics_view.get_dof_stiffnesses()
            kds = self._physics_view.get_dof_dampings()
            result_kps = self._backend_utils.move_data(
                kps[self._backend_utils.expand_dims(indices, 1), joint_indices], device=self._device
            )
            result_kds = self._backend_utils.move_data(
                kds[self._backend_utils.expand_dims(indices, 1), joint_indices], device=self._device
            )
            if clone:
                result_kps = self._backend_utils.clone_tensor(result_kps, device=self._device)
                result_kds = self._backend_utils.clone_tensor(result_kds, device=self._device)
            return result_kps, result_kds
        else:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            dof_types = self.get_dof_types()
            joint_indices = self._backend_utils.resolve_indices(joint_indices, self.num_dof, self._device)
            if joint_indices.numel() == 0:
                return None, None
            kps = self._backend_utils.create_zeros_tensor(
                shape=[indices.shape[0], joint_indices.shape[0]], dtype="float32", device=self._device
            )
            kds = self._backend_utils.create_zeros_tensor(
                shape=[indices.shape[0], joint_indices.shape[0]], dtype="float32", device=self._device
            )
            articulation_write_idx = 0
            for i in indices:
                dof_write_idx = 0
                for dof_index in joint_indices:
                    drive_type = (
                        "angular" if dof_types[dof_index] == omni.physics.tensors.DofType.Rotation else "linear"
                    )
                    prim = get_prim_at_path(self._dof_paths[i][dof_index])
                    if prim.HasAPI(UsdPhysics.DriveAPI):
                        drive = UsdPhysics.DriveAPI(prim, drive_type)
                    else:
                        drive = UsdPhysics.DriveAPI.Apply(prim, drive_type)
                    if drive.GetStiffnessAttr().Get() == 0.0 or drive_type == "linear":
                        kps[articulation_write_idx][dof_write_idx] = drive.GetStiffnessAttr().Get()
                    else:
                        kps[articulation_write_idx][dof_write_idx] = self._backend_utils.convert(
                            1.0 / isaacsim.utils.numpy.deg2rad(float(1.0 / drive.GetStiffnessAttr().Get())),
                            device=self._device,
                        )
                    if drive.GetDampingAttr().Get() == 0.0 or drive_type == "linear":
                        kds[articulation_write_idx][dof_write_idx] = drive.GetDampingAttr().Get()
                    else:
                        kds[articulation_write_idx][dof_write_idx] = self._backend_utils.convert(
                            1.0 / isaacsim.utils.numpy.deg2rad(float(1.0 / drive.GetDampingAttr().Get())),
                            device=self._device,
                        )
                    dof_write_idx += 1
                articulation_write_idx += 1
            return kps, kds
    
    def get_applied_actions(self, clone: bool = True) -> ArticulationActions:
        """Gets current applied actions in an ArticulationActions object.

        Args:
            clone (bool, optional): True to return clones of the internal buffers. Otherwise False. Defaults to True.

        Returns:
            ArticulationActions: current applied actions (i.e: current position targets and velocity targets)
        """
        if not self._is_initialized:
            carb.log_warn("ArticulationView needs to be initialized.")
            return None
        if not omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
            if self.num_dof == 0:
                return None
            # Check if _physics_sim_view is available before using it
            if self._physics_sim_view is not None:
                self._physics_sim_view.enable_warnings(False)
            joint_positions = self._physics_view.get_dof_position_targets()
            if clone:
                joint_positions = self._backend_utils.clone_tensor(joint_positions, device=self._device)
            joint_velocities = self._physics_view.get_dof_velocity_targets()
            if clone:
                joint_velocities = self._backend_utils.clone_tensor(joint_velocities, device=self._device)
            # Check if _physics_sim_view is available before using it
            if self._physics_sim_view is not None:
                self._physics_sim_view.enable_warnings(True)
            # TODO: implement the effort part
            return ArticulationActions(
                joint_positions=joint_positions,
                joint_velocities=joint_velocities,
                joint_efforts=None,
                joint_indices=None,
            )
        else:
            carb.log_warn("Physics Simulation View is not created yet in order to use get_applied_actions")
            return None

    def get_world_poses(
        self, env_indices: Optional[torch.Tensor] = None, clone: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = self._resolve_env_indices(env_indices)
        if self._physics_view is not None:
            with disable_warnings(self._physics_sim_view):
                poses = self._physics_view.get_root_transforms()[indices]
                poses = torch.unflatten(poses, 0, self.shape)
            if clone:
                poses = poses.clone()
            return poses[..., :3], poses[..., [6, 3, 4, 5]]
        else:
            pos, rot = super().get_world_poses(indices, clone)
            return pos.unflatten(0, self.shape), rot.unflatten(0, self.shape)

    def set_world_poses(
        self,
        positions: Optional[torch.Tensor] = None,
        orientations: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
    ) -> None:
        with disable_warnings(self._physics_sim_view):
            indices = self._resolve_env_indices(env_indices)
            
            # Safety checks to prevent GPU crash
            if self._physics_view is None:
                print("[OmniDrones] WARNING: set_world_poses called but _physics_view is None")
                return
                
            try:
                poses = self._physics_view.get_root_transforms()
                
                # Validate indices bounds
                if indices is not None:
                    max_index = poses.shape[0] - 1
                    if torch.any(indices > max_index) or torch.any(indices < 0):
                        print(f"[OmniDrones] ERROR: Invalid indices detected. Max valid index: {max_index}, indices range: {indices.min()}-{indices.max()}")
                        print(f"[OmniDrones] Poses shape: {poses.shape}, indices: {indices}")
                        # Clamp indices to valid range
                        indices = torch.clamp(indices, 0, max_index)
                        print(f"[OmniDrones] Clamped indices to valid range: {indices}")
                
                if positions is not None:
                    positions_reshaped = positions.reshape(-1, 3)
                    if indices.shape[0] != positions_reshaped.shape[0]:
                        print(f"[OmniDrones] WARNING: Shape mismatch - indices: {indices.shape}, positions: {positions_reshaped.shape}")
                    poses[indices, :3] = positions_reshaped
                    
                if orientations is not None:
                    orientations_reshaped = orientations.reshape(-1, 4)[:, [1, 2, 3, 0]]
                    if indices.shape[0] != orientations_reshaped.shape[0]:
                        print(f"[OmniDrones] WARNING: Shape mismatch - indices: {indices.shape}, orientations: {orientations_reshaped.shape}")
                    poses[indices, 3:] = orientations_reshaped
                
                print(f"[OmniDrones] set_world_poses: poses shape {poses.shape}, indices shape {indices.shape}")
                self._physics_view.set_root_transforms(poses, indices)
                
            except Exception as e:
                print(f"[OmniDrones] ERROR in set_world_poses: {e}")
                print(f"[OmniDrones] Debug info - poses shape: {poses.shape if 'poses' in locals() else 'N/A'}")
                print(f"[OmniDrones] Debug info - indices: {indices if 'indices' in locals() else 'N/A'}")
                raise e

    def get_velocities(
        self, env_indices: Optional[torch.Tensor] = None, clone: bool = True
    ) -> torch.Tensor:
        indices = self._resolve_env_indices(env_indices)
        return super().get_velocities(indices, clone).unflatten(0, self.shape)

    def set_velocities(
        self, velocities: torch.Tensor, env_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        indices = self._resolve_env_indices(env_indices)
        return super().set_velocities(velocities.reshape(-1, 6), indices)

    def get_joint_velocities(
        self, env_indices: Optional[torch.Tensor] = None, clone: bool = True
    ) -> torch.Tensor:
        indices = self._resolve_env_indices(env_indices)
        return (
            super().get_joint_velocities(indices, clone=clone).unflatten(0, self.shape)
        )

    def set_joint_velocities(
        self,
        velocities: Optional[torch.Tensor],
        env_indices: Optional[torch.Tensor] = None,
        joint_indices: Optional[torch.Tensor] = None,
    ) -> None:
        indices = self._resolve_env_indices(env_indices)
        super().set_joint_velocities(
            velocities.flatten(end_dim=-2), 
            indices,
            joint_indices
        )

    def set_joint_velocity_targets(
        self, 
        velocities: Optional[torch.Tensor], 
        env_indices: Optional[torch.Tensor] = None, 
        joint_indices: Optional[torch.Tensor] = None
    ) -> None:
        indices = self._resolve_env_indices(env_indices)
        super().set_joint_velocity_targets(
            velocities.flatten(end_dim=-2), 
            indices, 
            joint_indices
        )

    def get_joint_positions(
        self, env_indices: Optional[torch.Tensor] = None, clone: bool = True
    ) -> torch.Tensor:
        indices = self._resolve_env_indices(env_indices)
        return (
            super().get_joint_positions(indices, clone=clone).unflatten(0, self.shape)
        )

    def set_joint_positions(
        self,
        positions: Optional[torch.Tensor],
        env_indices: Optional[torch.Tensor] = None,
        joint_indices: Optional[torch.Tensor] = None,
    ) -> None:
        indices = self._resolve_env_indices(env_indices)
        super().set_joint_positions(
            positions.flatten(end_dim=-2), 
            indices,
            joint_indices
        )

    def set_joint_position_targets(
        self, 
        positions: Optional[torch.Tensor], 
        env_indices: Optional[torch.Tensor] = None, 
        joint_indices: Optional[torch.Tensor] = None,
    ) -> None:
        indices = self._resolve_env_indices(env_indices)
        super().set_joint_position_targets(
            positions.flatten(end_dim=-2), 
            indices,
            joint_indices
        )
    
    def set_joint_efforts(
        self, 
        efforts: Optional[torch.Tensor], 
        env_indices: Optional[torch.Tensor] = None, 
        joint_indices: Optional[torch.Tensor] = None
    ) -> None:
        indices = self._resolve_env_indices(env_indices)
        super().set_joint_efforts(
            efforts.flatten(end_dim=-2), 
            indices, 
            joint_indices
        )

    def get_dof_limits(self) -> torch.Tensor:
        return (
            super().get_dof_limits()
            .unflatten(0, self.shape)
            .to(self._device)
        )

    def get_body_masses(
        self, env_indices: Optional[torch.Tensor] = None, clone: bool = True
    ) -> torch.Tensor:
        indices = self._resolve_env_indices(env_indices)
        return super().get_body_masses(indices, clone=clone).unflatten(0, self.shape)

    def set_body_masses(
        self,
        values: torch.Tensor,
        env_indices: Optional[torch.Tensor] = None,
    ) -> None:
        indices = self._resolve_env_indices(env_indices)
        return super().set_body_masses(values.reshape(-1, self.num_bodies), indices)

    def get_force_sensor_forces(self, env_indices: Optional[torch.Tensor] = None, clone: bool = False) -> torch.Tensor:
        with disable_warnings(self._physics_sim_view):
            forces = torch.unflatten(self._physics_view.get_force_sensor_forces(), 0, self.shape)
        if clone:
            forces = forces.clone()
        if env_indices is not None:
            forces = forces[env_indices]
        return forces

    def _resolve_env_indices(self, env_indices: torch.Tensor):
        if not hasattr(self, "_all_indices"):
            self._all_indices = torch.arange(self.count, device=self._device)
            self.shape = self._all_indices.reshape(self.shape).shape
        if env_indices is not None:
            indices = self._all_indices.reshape(self.shape)[env_indices].flatten()
        else:
            indices = self._all_indices
        return indices

    def squeeze_(self, dim: int = None):
        self.shape = self._all_indices.reshape(self.shape).squeeze(dim).shape
        return self


class RigidPrimView(_RigidPrimView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: str = "rigid_prim_view",
        positions: Optional[torch.Tensor] = None,
        translations: Optional[torch.Tensor] = None,
        orientations: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
        visibilities: Optional[torch.Tensor] = None,
        reset_xform_properties: bool = True,
        masses: Optional[torch.Tensor] = None,
        densities: Optional[torch.Tensor] = None,
        linear_velocities: Optional[torch.Tensor] = None,
        angular_velocities: Optional[torch.Tensor] = None,
        track_contact_forces: bool = False,
        prepare_contact_sensors: bool = True,
        disable_stablization: bool = True,
        contact_filter_prim_paths_expr: Optional[List[str]] = (),
        shape: Tuple[int, ...] = (-1,),
    ) -> None:
        self.shape = shape
        super().__init__(
            prim_paths_expr,
            name,
            positions,
            translations,
            orientations,
            scales,
            visibilities,
            reset_xform_properties,
            masses,
            densities,
            linear_velocities,
            angular_velocities,
            track_contact_forces,
            prepare_contact_sensors,
            disable_stablization,
            contact_filter_prim_paths_expr,
        )

    @require_sim_initialized
    def initialize(self, physics_sim_view: omni.physics.tensors.SimulationView = None):
        super().initialize(physics_sim_view)
        self.shape = torch.arange(self.count).reshape(self.shape).shape
        return self

    def get_world_poses(
        self, env_indices: Optional[torch.Tensor] = None, clone: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = self._resolve_env_indices(env_indices)
        pos, rot = super().get_world_poses(indices, clone)
        return pos.unflatten(0, self.shape), rot.unflatten(0, self.shape)

    def set_world_poses(
        self,
        positions: Optional[torch.Tensor] = None,
        orientations: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
    ) -> None:
        with disable_warnings(self._physics_sim_view):
            indices = self._resolve_env_indices(env_indices)
            
            # Safety checks to prevent GPU crash
            if self._physics_view is None:
                print("[OmniDrones] WARNING: set_world_poses called but _physics_view is None")
                return
                
            try:
                poses = self._physics_view.get_transforms()
                
                # Validate indices bounds
                if indices is not None:
                    max_index = poses.shape[0] - 1
                    if torch.any(indices > max_index) or torch.any(indices < 0):
                        print(f"[OmniDrones] ERROR: Invalid indices detected. Max valid index: {max_index}, indices range: {indices.min()}-{indices.max()}")
                        print(f"[OmniDrones] Poses shape: {poses.shape}, indices: {indices}")
                        # Clamp indices to valid range
                        indices = torch.clamp(indices, 0, max_index)
                        print(f"[OmniDrones] Clamped indices to valid range: {indices}")
                
                if positions is not None:
                    positions_reshaped = positions.reshape(-1, 3)
                    if indices.shape[0] != positions_reshaped.shape[0]:
                        print(f"[OmniDrones] WARNING: Shape mismatch - indices: {indices.shape}, positions: {positions_reshaped.shape}")
                    poses[indices, :3] = positions_reshaped
                    
                if orientations is not None:
                    orientations_reshaped = orientations.reshape(-1, 4)[:, [1, 2, 3, 0]]
                    if indices.shape[0] != orientations_reshaped.shape[0]:
                        print(f"[OmniDrones] WARNING: Shape mismatch - indices: {indices.shape}, orientations: {orientations_reshaped.shape}")
                    poses[indices, 3:] = orientations_reshaped
                
                print(f"[OmniDrones] RigidPrimView set_world_poses: poses shape {poses.shape}, indices shape {indices.shape}")
                self._physics_view.set_transforms(poses, indices)
                
            except Exception as e:
                print(f"[OmniDrones] ERROR in RigidPrimView set_world_poses: {e}")
                print(f"[OmniDrones] Debug info - poses shape: {poses.shape if 'poses' in locals() else 'N/A'}")
                print(f"[OmniDrones] Debug info - indices: {indices if 'indices' in locals() else 'N/A'}")
                raise e

    def get_velocities(
        self, env_indices: Optional[torch.Tensor] = None, clone: bool = True
    ) -> torch.Tensor:
        indices = self._resolve_env_indices(env_indices)
        return super().get_velocities(indices, clone).unflatten(0, self.shape)

    def set_velocities(
        self, velocities: torch.Tensor, env_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        indices = self._resolve_env_indices(env_indices)
        return super().set_velocities(velocities.reshape(-1, 6), indices)

    def get_net_contact_forces(
        self,
        env_indices: Optional[torch.Tensor] = None,
        clone: bool = False,
        dt: float = 1,
    ) -> torch.Tensor:
        indices = self._resolve_env_indices(env_indices)
        return (
            super().get_net_contact_forces(indices, clone, dt).unflatten(0, self.shape)
        )

    def get_contact_force_matrix(
        self, 
        env_indices: Optional[torch.Tensor] = None, 
        clone: bool = True, 
        dt: float = 1
    ) -> torch.Tensor:
        indices = self._resolve_env_indices(env_indices)
        return super().get_contact_force_matrix(indices, clone, dt).unflatten(0, self.shape)

    def get_masses(
        self, 
        env_indices: Optional[torch.Tensor] = None, 
        clone: bool = True
    ) -> torch.Tensor:
        indices = self._resolve_env_indices(env_indices)
        if not omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
            current_values = self._backend_utils.move_data(self._physics_view.get_masses(), self._device)
            masses = current_values[indices]
            if clone:
                masses = self._backend_utils.clone_tensor(masses, device=self._device)
        else:
            masses = self._backend_utils.create_zeros_tensor([indices.shape[0]], dtype="float32", device=self._device)
            write_idx = 0
            for i in indices:
                if self._mass_apis[i.tolist()] is None:
                    if self._prims[i.tolist()].HasAPI(UsdPhysics.MassAPI):
                        self._mass_apis[i.tolist()] = UsdPhysics.MassAPI(self._prims[i.tolist()])
                    else:
                        self._mass_apis[i.tolist()] = UsdPhysics.MassAPI.Apply(self._prims[i.tolist()])
                masses[write_idx] = self._backend_utils.create_tensor_from_list(
                    self._mass_apis[i.tolist()].GetMassAttr().Get(), dtype="float32", device=self._device
                )
                write_idx += 1
        return masses.reshape(-1, *self.shape[1:], 1)
    
    def set_masses(
        self, 
        masses: torch.Tensor, 
        env_indices: Optional[torch.Tensor] = None
    ) -> None:
        indices = self._resolve_env_indices(env_indices).cpu()
        masses = masses.reshape(-1, 1)
        if not omni.timeline.get_timeline_interface().is_stopped() and self._physics_view is not None:
            data = self._backend_utils.clone_tensor(self._physics_view.get_masses(), device="cpu")
            data[indices] = self._backend_utils.move_data(masses, device="cpu")
            self._physics_view.set_masses(data, indices)
        else:
            indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
            read_idx = 0
            for i in indices:
                if self._mass_apis[i.tolist()] is None:
                    if self._prims[i.tolist()].HasAPI(UsdPhysics.MassAPI):
                        self._mass_apis[i.tolist()] = UsdPhysics.MassAPI(self._prims[i.tolist()])
                    else:
                        self._mass_apis[i.tolist()] = UsdPhysics.MassAPI.Apply(self._prims[i.tolist()])
                self._mass_apis[i.tolist()].GetMassAttr().Set(masses[read_idx].tolist())
                read_idx += 1
            return

    def get_coms(
        self, 
        env_indices: Optional[torch.Tensor] = None, 
        clone: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = self._resolve_env_indices(env_indices)
        positions, orientations = super().get_coms(indices, clone)
        return positions.unflatten(0, self.shape), orientations.unflatten(0, self.shape)
    
    def set_coms(
        self, 
        positions: torch.Tensor = None, 
        # orientations: torch.Tensor = None, 
        env_indices: torch.Tensor = None
    ) -> None:
        # TODO@btx0424 fix orientations
        indices = self._resolve_env_indices(env_indices)
        return super().set_coms(positions.reshape(-1, 1, 3), None, indices)
    
    def get_inertias(
        self, 
        env_indices: Optional[torch.Tensor]=None, 
        clone: bool=True
    ) -> torch.Tensor:
        indices = self._resolve_env_indices(env_indices)
        return super().get_inertias(indices, clone).unflatten(0, self.shape)
    
    def set_inertias(
        self, 
        values: torch.Tensor, 
        env_indices: Optional[torch.Tensor]=None
    ):
        indices = self._resolve_env_indices(env_indices)
        return super().set_inertias(values.reshape(-1, 9), indices)

    def _resolve_env_indices(self, env_indices: torch.Tensor):
        if not hasattr(self, "_all_indices"):
            self._all_indices = torch.arange(self.count, device=self._device)
            self.shape = self._all_indices.reshape(self.shape).shape
        if env_indices is not None:
            indices = self._all_indices.reshape(self.shape)[env_indices].flatten()
        else:
            indices = self._all_indices
        return indices

    def squeeze_(self, dim: int = None):
        self.shape = self._all_indices.reshape(self.shape).squeeze(dim).shape
        return self


@contextmanager
def disable_warnings(physics_sim_view):
    try:
        physics_sim_view.enable_warnings(False)
        yield
    finally:
        physics_sim_view.enable_warnings(True)
