from yacs.config import CfgNode as CN


_C = CN()

_C.CHARACTER_GENERATION = CN()
_C.CHARACTER_GENERATION.UPSCALER = "rrdbnet"

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
