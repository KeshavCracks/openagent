"""Quality gates + slash commands (progressive-disclosure kernel)."""
from plugins.quality_gates.engine import (
    DEFINITIONS,
    GateBlock,
    handle_slash,
    validate_output,
    validate_pre_action,
)
