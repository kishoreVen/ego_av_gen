"""Control playbooks for quality control pipeline."""

from typing import Dict, Type

from quality_control.playbook import Playbook
from quality_control.control_playbook.global_solve import GlobalSolvePlaybook
from quality_control.control_playbook.swarm_solve import SwarmSolvePlaybook
from quality_control.control_playbook.multi_stage_solve import (
    MultiStageSolvePlaybook,
)
from quality_control.control_playbook.batched_vision_solve import (
    BatchedVisionSolvePlaybook,
)

PLAYBOOK_REGISTRY: Dict[str, Type[Playbook]] = {
    "GlobalSolve": GlobalSolvePlaybook,
    "SwarmSolve": SwarmSolvePlaybook,
    "MultiStageSolve": MultiStageSolvePlaybook,
    "BatchedVisionSolve": BatchedVisionSolvePlaybook,
}
