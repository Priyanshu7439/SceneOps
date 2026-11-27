from sceneops.simulator.deterioration import (
    DeteriorationEvent,
    PatientChanges,
)
from sceneops.simulator.patient_profile_generator import (
    PatientProfileGenerator,
)
from sceneops.simulator.patients import InjuryType, Patient
from sceneops.simulator.resources import Ambulance
from sceneops.simulator.scenario import ScenarioDefinition
from sceneops.simulator.scenario_config import ScenarioConfig
from sceneops.simulator.scenario_event_generator import (
    ScenarioEventGenerator,
)
from sceneops.simulator.scenario_validator import ScenarioValidator

SCENARIO_DURATION_SECONDS = 600
PATIENTS_PER_CATEGORY = 4
AMBULANCE_COUNT = 5
DEFAULT_SCENARIO_INDICES = (1, 2, 3, 4, 5)


def create_mass_casualty_scenario(
    scenario_index: int = 1,
) -> ScenarioDefinition:
    """
    Create a deterministic synthetic 20-patient training scenario.

    Scenario index 1 preserves the original MCI-001 scenario.
    Higher indexes introduce controlled variations while preserving
    the five initial triage archetypes.
    """
    if isinstance(scenario_index, bool) or not isinstance(scenario_index, int):
        raise TypeError("scenario_index must be an integer.")

    if scenario_index < 1:
        raise ValueError("scenario_index must be >= 1.")

    scenario_id = f"MCI-{scenario_index:03d}"

    # Controlled variation. Index 1 produces zero offsets.
    variation = scenario_index - 1
    respiratory_offset = variation % 3
    demographic_offset = (variation * 7) % 55
    injury_offset = variation % len(InjuryType)
    event_offset = (variation % 5) * 30

    # Four patient archetypes per initial triage category.
    profiles = [
        # GREEN: ambulatory patients.
        *[
            {
                "ambulatory": True,
                "respiratory_rate": rate + respiratory_offset,
                "radial_pulse_present": True,
                "follows_commands": True,
            }
            for rate in (20, 18, 22, 24)
        ],
        # YELLOW: non-ambulatory, initially stable.
        *[
            {
                "ambulatory": False,
                "respiratory_rate": 20 + respiratory_offset,
                "radial_pulse_present": True,
                "follows_commands": True,
            }
            for _ in range(PATIENTS_PER_CATEGORY)
        ],
        # RED: abnormal respiratory rate.
        *[
            {
                "ambulatory": False,
                "respiratory_rate": 35 + respiratory_offset,
                "radial_pulse_present": True,
                "follows_commands": True,
            }
            for _ in range(PATIENTS_PER_CATEGORY)
        ],
        # BLACK: remain apneic after airway repositioning.
        *[
            {
                "ambulatory": False,
                "respiratory_rate": 0,
                "airway_opened": True,
                "radial_pulse_present": False,
                "follows_commands": False,
            }
            for _ in range(PATIENTS_PER_CATEGORY)
        ],
        # PENDING: respiratory assessment not recorded.
        *[
            {
                "ambulatory": False,
                "respiratory_rate": None,
                "radial_pulse_present": True,
                "follows_commands": True,
            }
            for _ in range(PATIENTS_PER_CATEGORY)
        ],
    ]

    injury_types = list(InjuryType)

    patients = [
        Patient(
            patient_id=f"P{index + 1:03d}",
            age=20 + (index * 3 + demographic_offset) % 55,
            injury_type=injury_types[(index + injury_offset) % len(injury_types)],
            major_hemorrhage=False,
            **profile,
        )
        for index, profile in enumerate(profiles)
    ]

    ambulances = [
        Ambulance(ambulance_id=f"A{index + 1:03d}") for index in range(AMBULANCE_COUNT)
    ]

    deterioration_events = [
        DeteriorationEvent(
            event_id="EV001",
            patient_id="P005",
            at_seconds=120 + event_offset,
            changes=PatientChanges(respiratory_rate=35),
        ),
        DeteriorationEvent(
            event_id="EV002",
            patient_id="P006",
            at_seconds=240 + event_offset,
            changes=PatientChanges(follows_commands=False),
        ),
        DeteriorationEvent(
            event_id="EV003",
            patient_id="P011",
            at_seconds=360 + event_offset,
            changes=PatientChanges(follows_commands=False),
        ),
    ]

    # Keep scenario metadata aligned with the configured event count.
    if scenario_index == 1:
        title = "20-Patient Mass Casualty Exercise"
        description = (
            "Synthetic educational scenario with 20 patients, "
            "five ambulances and three timed deterioration events."
        )
    else:
        title = f"20-Patient Mass Casualty Exercise {scenario_index}"
        description = (
            "Synthetic educational scenario with 20 patients, "
            "five ambulances and three timed deterioration events. "
            f"Deterministic variation index: {scenario_index}."
        )

    return ScenarioDefinition(
        scenario_id=scenario_id,
        title=title,
        description=description,
        duration_seconds=SCENARIO_DURATION_SECONDS,
        patients=patients,
        ambulances=ambulances,
        deterioration_events=deterioration_events,
    )


def create_hybrid_scenario(
    config: ScenarioConfig,
) -> ScenarioDefinition:
    """
    Generate a reproducible scenario using configuration-based
    structure and seed-based patient and event variations.

    The existing indexed scenario factory remains independent.
    """
    if not isinstance(config, ScenarioConfig):
        raise TypeError("config must be a ScenarioConfig instance.")

    patient_generator = PatientProfileGenerator()
    event_generator = ScenarioEventGenerator()
    validator = ScenarioValidator()

    # Generate patient profiles.
    patients = patient_generator.generate(config)

    # Generate deterioration events for these patients.
    deterioration_events = event_generator.generate(
        config,
        patients,
    )

    # Generate ambulance resources.
    ambulances = [
        Ambulance(ambulance_id=f"A{index:03d}")
        for index in range(1, config.ambulance_count + 1)
    ]

    # Assemble the scenario.
    scenario = ScenarioDefinition(
        scenario_id=config.scenario_id,
        title=f"Hybrid Mass Casualty Exercise - {config.scenario_id}",
        description=(
            "Synthetic educational mass-casualty scenario "
            "generated using configuration-based structure "
            "and deterministic seed-based variations."
        ),
        duration_seconds=config.duration_seconds,
        patients=patients,
        ambulances=ambulances,
        deterioration_events=deterioration_events,
    )

    # Validate before returning.
    validator.validate(config, scenario)

    return scenario
