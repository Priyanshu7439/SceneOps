import asyncio

from sceneops.agents.contracts import AgentRequest, AgentTask, TemporalObservation
from sceneops.agents.temporal_scene_narration import TemporalSceneNarrationAgent


class CapturingBackend:
    def __init__(self):
        self.system_prompt = ""
        self.user_prompt = ""

    async def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return (
            "At simulation second 0, Patient P005: respiratory rate is 20 "
            "breaths per minute. "
            "At simulation second 60, Patient P005: respiratory rate is 20 "
            "breaths per minute. "
            "At simulation second 120, Patient P005: respiratory rate is 35 "
            "breaths per minute."
        )


def test_temporal_agent_prompt_requires_one_measurement_per_exact_timestamp():
    backend = CapturingBackend()
    request = AgentRequest(
        task=AgentTask.TEMPORAL_SCENE_NARRATION,
        user_message="Summarize the chronological observations.",
        temporal_observations=(
            TemporalObservation(
                patient_id="P005",
                observation_type="respiratory_rate",
                value="20",
                observed_at_seconds=0,
            ),
            TemporalObservation(
                patient_id="P005",
                observation_type="respiratory_rate",
                value="20",
                observed_at_seconds=60,
            ),
            TemporalObservation(
                patient_id="P005",
                observation_type="respiratory_rate",
                value="35",
                observed_at_seconds=120,
            ),
        ),
    )

    response = asyncio.run(TemporalSceneNarrationAgent(backend).generate(request))

    assert "exactly one separate sentence for each supplied observation" in backend.system_prompt
    assert "NEVER" in backend.system_prompt
    assert "between simulation seconds X and Y" in backend.system_prompt
    assert "Simulation second 0: Patient P005, respiratory_rate = 20" in backend.system_prompt
    assert "Simulation second 60: Patient P005, respiratory_rate = 20" in backend.system_prompt
    assert "Simulation second 120: Patient P005, respiratory_rate = 35" in backend.system_prompt
    assert "Canonical rendering targets" in backend.system_prompt
    assert "Follow the system instructions exactly" in backend.user_prompt
    assert response.content.count("At simulation second") == 3
