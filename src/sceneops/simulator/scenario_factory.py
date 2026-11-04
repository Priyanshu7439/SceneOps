
from sceneops.simulator.deterioration import (
    DeteriorationEvent,
    PatientChanges,
)
from sceneops.simulator.patients import InjuryType, Patient
from sceneops.simulator.resources import Ambulance
from sceneops.simulator.scenario import ScenarioDefinition


def create_mass_casualty_scenario() -> ScenarioDefinition:
    """Create a deterministic 20-patient synthetic training scenario."""

    profiles = [
        # 4 GREEN: ambulatory patients
        {
            "ambulatory": True,
            "respiratory_rate": 20,
            "radial_pulse_present": True,
            "follows_commands": True,
        },
        {
            "ambulatory": True,
            "respiratory_rate": 18,
            "radial_pulse_present": True,
            "follows_commands": True,
        },
        {
            "ambulatory": True,
            "respiratory_rate": 22,
            "radial_pulse_present": True,
            "follows_commands": True,
        },
        {
            "ambulatory": True,
            "respiratory_rate": 24,
            "radial_pulse_present": True,
            "follows_commands": True,
        },

        # 4 YELLOW: non-ambulatory, stable initial observations
        *[
            {
                "ambulatory": False,
                "respiratory_rate": 20,
                "radial_pulse_present": True,
                "follows_commands": True,
            }
            for _ in range(4)
        ],

        # 4 RED: abnormal respiratory rate
        *[
            {
                "ambulatory": False,
                "respiratory_rate": 35,
                "radial_pulse_present": True,
                "follows_commands": True,
            }
            for _ in range(4)
        ],

        # 4 BLACK: remain apneic after airway repositioning
        *[
            {
                "ambulatory": False,
                "respiratory_rate": 0,
                "airway_opened": True,
                "radial_pulse_present": False,
                "follows_commands": False,
            }
            for _ in range(4)
        ],

        # 4 PENDING: respiratory assessment not yet recorded
        *[
            {
                "ambulatory": False,
                "respiratory_rate": None,
                "radial_pulse_present": True,
                "follows_commands": True,
            }
            for _ in range(4)
        ],
    ]

    injury_types = list(InjuryType)

    patients = [
        Patient(
            patient_id=f"P{index + 1:03d}",
            age=20 + (index * 3) % 55,
            injury_type=injury_types[index % len(injury_types)],
            major_hemorrhage=False,
            **profile,
        )
        for index, profile in enumerate(profiles)
    ]

    ambulances = [
        Ambulance(ambulance_id=f"A{index + 1:03d}")
        for index in range(5)
    ]

    deterioration_events = [
        DeteriorationEvent(
            event_id="EV001",
            patient_id="P005",
            at_seconds=120,
            changes=PatientChanges(respiratory_rate=35),
        ),
        DeteriorationEvent(
            event_id="EV002",
            patient_id="P006",
            at_seconds=240,
            changes=PatientChanges(follows_commands=False),
        ),
    ]

    return ScenarioDefinition(
        scenario_id="MCI-001",
        title="20-Patient Mass Casualty Exercise",
        description=(
            "Synthetic educational scenario with 20 patients, "
            "five ambulances and two timed deterioration events."
        ),
        duration_seconds=600,
        patients=patients,
        ambulances=ambulances,
        deterioration_events=deterioration_events,
    )
