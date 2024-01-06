print("hello")
from jaxtyping import install_import_hook
# Plus any one of the following:

# decorate `@jaxtyped(typechecker=beartype.beartype)`
with install_import_hook("mamba", "beartype.beartype"):
    from .mamba import *