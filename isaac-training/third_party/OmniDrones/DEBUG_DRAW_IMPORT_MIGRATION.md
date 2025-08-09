# Robust `debug_draw` Import Pattern for IsaacSim/IsaacLab Migration

## Motivation

The `omni.isaac.debug_draw` module and related imports have been deprecated and/or moved in recent IsaacSim and IsaacLab releases. Direct imports from the old path can cause `ModuleNotFoundError` or break compatibility with new versions. To ensure your codebase works across all supported IsaacSim/IsaacLab versions, a robust try/fallback import pattern is required.

## Problem

- **Old (deprecated):**
  ```python
  from omni.isaac.debug_draw import _debug_draw
  ```
- **New (IsaacSim/IsaacLab):**
  ```python
  from isaacsim.debug_draw import _debug_draw
  # or
  from isaacsim.util.debug_draw._debug_draw import _debug_draw
  ```
- **If none are available:**
  - You need a minimal fallback to avoid runtime errors.

## Solution: Robust Import Pattern

Use the following pattern at the top of any file that needs `_debug_draw`:

```python
try:
    from isaacsim.debug_draw import _debug_draw
except ImportError:
    try:
        from isaacsim.util.debug_draw._debug_draw import _debug_draw
    except ImportError:
        try:
            from omni.isaac.debug_draw import _debug_draw
        except ImportError:
            class _MockDebugDraw:
                @staticmethod
                def acquire_debug_draw_interface():
                    return None
            _debug_draw = _MockDebugDraw()
```

## Usage Example

```python
self.draw = _debug_draw.acquire_debug_draw_interface()
if self.draw is not None:
    self.draw.clear_lines()
    # ... other debug drawing ...
```

## Why This Works
- **Future-proof:** Works with both new and legacy IsaacSim/IsaacLab versions.
- **No runtime errors:** If no debug_draw is available, code will not crash.
- **Consistent:** Use this pattern everywhere you need debug drawing.

## Migration Checklist
- [x] Replace all direct `omni.isaac.debug_draw` imports with the robust pattern above.
- [x] Update all files that use debug drawing (see codebase-wide refactor).
- [x] Add this documentation file for future maintainers.

## References
- IsaacSim/IsaacLab migration guides
- IndiDrones and IsaacLab codebase best practices
