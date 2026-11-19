from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from sceneops.agents.contracts import SceneFact
from sceneops.agents.scene_fact_builder import SceneFactBuilder
from sceneops.simulator.observations import ObservationType
from sceneops.simulator.scenario_builder import ScenarioBuilder
from sceneops.training.dataset import ChatMessage
from sceneops.training.scene_narration_dataset import SceneNarrationExample
from sceneops.training.splitting import split_by_scenario
from sceneops.training.temporal_scenario_generator import (
    TemporalScenarioGenerator,
)
from sceneops.training.temporal_scene_narration_dataset import (
    TemporalSceneNarrationExample,
)
from sceneops.training.temporal_scene_narration_validator import (
    TemporalSceneNarrationValidator,
)

TEMPORAL_SCENE_GENERATOR_V3_VERSION = "3.1.0"
FULL_DATASET_GENERATOR_VERSION = "3.1.0"
MINIMUM_DATASET_EXAMPLES = 5500
DEFAULT_SCENARIO_COUNT = 320

# Only observations explicitly supported by SceneFactBuilder and the engine.
TRAINING_OBSERVATION_TYPES = (
    ObservationType.RESPIRATORY_RATE,
    ObservationType.RADIAL_PULSE,
    ObservationType.CAPILLARY_REFILL,
    ObservationType.MENTAL_STATUS,
    ObservationType.AMBULATORY_STATUS,
    ObservationType.VISIBLE_HEMORRHAGE,
    ObservationType.FOLLOWS_COMMANDS,
)

# Kept for compatibility with the original event-level V3 generator.
FIELD_OBSERVATION_MAP = {
    "respiratory_rate": ObservationType.RESPIRATORY_RATE,
    "ambulatory": ObservationType.AMBULATORY_STATUS,
    "radial_pulse_present": ObservationType.RADIAL_PULSE,
    "follows_commands": ObservationType.FOLLOWS_COMMANDS,
    "major_hemorrhage": ObservationType.VISIBLE_HEMORRHAGE,
    "mental_status": ObservationType.MENTAL_STATUS,
}

type TrainingExample = SceneNarrationExample | TemporalSceneNarrationExample


class TemporalSceneNarrationDatasetV3Generator:
    """Generate event-level and full-panel scene narration datasets."""

    def __init__(self) -> None:
        self.scenario_generator = TemporalScenarioGenerator()

    def generate(
        self,
        seeds: tuple[int, ...] = tuple(range(100)),
    ) -> list[TemporalSceneNarrationExample]:
        """Generate the legacy event-level examples retained for regression."""
        self._validate_seeds(seeds)
        examples: list[TemporalSceneNarrationExample] = []

        for seed in seeds:
            scenario = self.scenario_generator.generate(seed)
            engine = ScenarioBuilder.build(scenario)
            engine.start()
            events_by_time = self._group_events_by_time(scenario.deterioration_events)
            current_time = engine.clock.elapsed_seconds

            for event_time in sorted(events_by_time):
                events = events_by_time[event_time]
                if event_time <= current_time:
                    raise ValueError(
                        "Temporal events must occur after the current timestamp."
                    )
                targets = self._get_event_targets(events)
                initial_observations = {
                    target: engine.observe_patient(*target) for target in targets
                }
                engine.advance_time(seconds=event_time - current_time)
                current_time = engine.clock.elapsed_seconds

                for target in targets:
                    patient_id, observation_type = target
                    follow_up = engine.observe_patient(patient_id, observation_type)
                    initial = initial_observations[target]
                    example = self._build_legacy_example(
                        scenario_id=scenario.scenario_id,
                        patient_id=patient_id,
                        observation_type=observation_type,
                        initial_observation=initial,
                        follow_up_observation=follow_up,
                        event_time=event_time,
                    )
                    examples.append(example)

        return examples

    def generate_full_dataset(
        self,
        seeds: tuple[int, ...] = tuple(range(DEFAULT_SCENARIO_COUNT)),
    ) -> list[TrainingExample]:
        """Generate one static and one full-panel temporal example per patient."""
        self._validate_seeds(seeds)
        examples: list[TrainingExample] = []

        for seed in seeds:
            scenario = self.scenario_generator.generate_diverse(seed)
            engine = ScenarioBuilder.build(scenario)
            engine.start()

            initial_observations = {
                patient_id: self._observe_patient_panel(engine, patient_id)
                for patient_id in engine.patients
            }

            current_time = engine.clock.elapsed_seconds
            for event_time in sorted(
                {event.at_seconds for event in scenario.deterioration_events}
            ):
                if event_time <= current_time:
                    raise ValueError(
                        f"Scenario {scenario.scenario_id} contains a non-future event."
                    )
                engine.advance_time(seconds=event_time - current_time)
                current_time = engine.clock.elapsed_seconds

            if current_time < scenario.duration_seconds:
                engine.advance_time(seconds=scenario.duration_seconds - current_time)

            final_observations = {
                patient_id: self._observe_patient_panel(engine, patient_id)
                for patient_id in engine.patients
            }

            for patient_id in engine.patients:
                initial = initial_observations[patient_id]
                final = final_observations[patient_id]
                initial_facts = SceneFactBuilder.build(initial)
                final_facts = SceneFactBuilder.build(final)

                if not initial_facts or not final_facts:
                    raise ValueError(
                        f"Patient {patient_id} in {scenario.scenario_id} "
                        "has no approved learner observations."
                    )

                initial_types = {fact.fact for fact in initial_facts}
                final_types = {fact.fact for fact in final_facts}
                if initial_types != final_types:
                    raise ValueError(
                        f"Patient {patient_id} in {scenario.scenario_id} "
                        "has inconsistent initial/final observation coverage."
                    )

                examples.append(
                    self._build_scene_example(
                        scenario_id=scenario.scenario_id,
                        patient_id=patient_id,
                        observed_at_seconds=scenario.duration_seconds,
                        facts=final_facts,
                    )
                )
                examples.append(
                    self._build_full_panel_temporal_example(
                        scenario_id=scenario.scenario_id,
                        patient_id=patient_id,
                        initial_facts=initial_facts,
                        follow_up_facts=final_facts,
                        initial_time=0,
                        follow_up_time=scenario.duration_seconds,
                    )
                )

        return examples

    @staticmethod
    def _validate_seeds(seeds: tuple[int, ...]) -> None:
        if not seeds:
            raise ValueError("At least one seed is required.")
        if len(seeds) != len(set(seeds)):
            raise ValueError("Seeds must be unique.")
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in seeds
        ):
            raise ValueError("Every seed must be a non-negative integer.")

    @staticmethod
    def _observe_patient_panel(engine, patient_id: str):
        observations = []
        for observation_type in TRAINING_OBSERVATION_TYPES:
            try:
                observations.append(
                    engine.observe_patient(patient_id, observation_type)
                )
            except ValueError as exc:
                # Missing values (e.g. an unrecorded respiratory rate) are
                # omitted; unsupported workflows must not be fabricated.
                if "no recorded value" not in str(exc):
                    raise
        return observations

    @staticmethod
    def _build_scene_example(
        *,
        scenario_id: str,
        patient_id: str,
        observed_at_seconds: int,
        facts: tuple[SceneFact, ...],
    ) -> SceneNarrationExample:
        labels_and_values = [
            TemporalSceneNarrationDatasetV3Generator._fact_label_value(fact)
            for fact in facts
        ]
        observation_text = "; ".join(
            f"{label}: {value}" for label, value in labels_and_values
        )
        system_prompt = (
            "You narrate a synthetic educational mass-casualty training scene. "
            "Use only learner-recorded observations supplied in the user message. "
            "Do not infer causes, injuries, symptoms, diagnoses, triage categories, "
            "or events. Do not reveal hidden simulator state."
        )
        user_prompt = (
            f"Summarize the learner-recorded observations for patient {patient_id} "
            f"at simulation second {observed_at_seconds}: {observation_text}."
        )
        narration = (
            f"At simulation second {observed_at_seconds}, the learner recorded "
            f"the following observations for patient {patient_id}: "
            f"{observation_text}."
        )
        return SceneNarrationExample(
            example_id=f"{scenario_id}-{patient_id}-scene-final",
            scenario_id=scenario_id,
            patient_id=patient_id,
            generator_version=FULL_DATASET_GENERATOR_VERSION,
            source_facts=facts,
            messages=(
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_prompt),
                ChatMessage(role="assistant", content=narration),
            ),
        )

    @staticmethod
    def _build_full_panel_temporal_example(
        *,
        scenario_id: str,
        patient_id: str,
        initial_facts: tuple[SceneFact, ...],
        follow_up_facts: tuple[SceneFact, ...],
        initial_time: int,
        follow_up_time: int,
    ) -> TemporalSceneNarrationExample:
        initial_by_type = {
            fact.fact.rsplit(":", maxsplit=1)[-1]: fact for fact in initial_facts
        }
        follow_up_by_type = {
            fact.fact.rsplit(":", maxsplit=1)[-1]: fact for fact in follow_up_facts
        }
        if initial_by_type.keys() != follow_up_by_type.keys():
            raise ValueError("Initial and follow-up observation types must match.")

        observation_types = tuple(ObservationType(name) for name in initial_by_type)
        primary_type = observation_types[0]

        initial_items = [
            TemporalSceneNarrationDatasetV3Generator._fact_label_value(fact)
            for fact in initial_facts
        ]
        follow_up_items = [
            TemporalSceneNarrationDatasetV3Generator._fact_label_value(fact)
            for fact in follow_up_facts
        ]
        initial_text = "; ".join(f"{label}: {value}" for label, value in initial_items)
        follow_up_text = "; ".join(
            f"{label}: {value}" for label, value in follow_up_items
        )

        changed = []
        unchanged = []
        for name, initial_fact in initial_by_type.items():
            initial_value = TemporalSceneNarrationDatasetV3Generator._fact_value(
                initial_fact
            )
            follow_up_value = TemporalSceneNarrationDatasetV3Generator._fact_value(
                follow_up_by_type[name]
            )
            label = name.replace("_", " ")
            if initial_value == follow_up_value:
                unchanged.append(f"{label} remained {initial_value}")
            else:
                changed.append(
                    f"{label} changed from {initial_value} to {follow_up_value}"
                )

        comparison = []
        if changed:
            comparison.append("Observed changes: " + "; ".join(changed))
        if unchanged:
            comparison.append("Unchanged observations: " + "; ".join(unchanged))
        comparison_text = ". ".join(comparison)

        system_prompt = (
            "You narrate a synthetic educational mass-casualty training scene. "
            "Use only learner-recorded observations supplied in the user message. "
            "Compare the two observation snapshots accurately. State which "
            "observations changed and which remained stable. Do not infer causes, "
            "injuries, symptoms, diagnoses, triage decisions, or hidden events."
        )
        user_prompt = (
            f"Compare learner-recorded observations for patient {patient_id}. "
            f"Initial observations at simulation second {initial_time}: "
            f"{initial_text}. Follow-up observations at simulation second "
            f"{follow_up_time}: {follow_up_text}."
        )
        narration = (
            f"For patient {patient_id}, at simulation second {initial_time}, "
            f"the learner recorded: {initial_text}. At simulation second "
            f"{follow_up_time}, the learner recorded: {follow_up_text}. "
            f"{comparison_text}."
        )

        return TemporalSceneNarrationExample(
            example_id=f"{scenario_id}-{patient_id}-temporal-full-panel",
            scenario_id=scenario_id,
            patient_id=patient_id,
            observation_type=primary_type,
            task="temporal_scene_narration",
            observation_types=observation_types,
            generator_version=FULL_DATASET_GENERATOR_VERSION,
            initial_facts=initial_facts,
            follow_up_facts=follow_up_facts,
            initial_observed_at_seconds=initial_time,
            follow_up_observed_at_seconds=follow_up_time,
            messages=(
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_prompt),
                ChatMessage(role="assistant", content=narration),
            ),
        )

    @staticmethod
    def _fact_label_value(fact: SceneFact) -> tuple[str, str]:
        label = fact.fact.rsplit(":", maxsplit=1)[-1].replace("_", " ")
        return label, TemporalSceneNarrationDatasetV3Generator._fact_value(fact)

    @staticmethod
    def _fact_value(fact: SceneFact) -> str:
        return fact.value.split(" (observed at simulation second ", maxsplit=1)[0]

    @staticmethod
    def _group_events_by_time(events):
        grouped = {}
        for event in events:
            grouped.setdefault(event.at_seconds, []).append(event)
        return grouped

    @staticmethod
    def _get_event_targets(events):
        targets = set()
        for event in events:
            for field_name, new_value in event.changes.model_dump(
                exclude_unset=True
            ).items():
                observation_type = FIELD_OBSERVATION_MAP.get(field_name)
                if observation_type is None:
                    raise ValueError(
                        f"No learner observation mapping for changed field: "
                        f"{field_name}."
                    )
                if new_value is None:
                    raise ValueError(
                        f"Cannot observe {field_name} after a change to None."
                    )
                targets.add((event.patient_id, observation_type))
        if not targets:
            raise ValueError("No observable event changes were found.")
        return sorted(targets, key=lambda item: (item[0], item[1].value))

    @staticmethod
    def _build_legacy_example(
        *,
        scenario_id: str,
        patient_id: str,
        observation_type: ObservationType,
        initial_observation,
        follow_up_observation,
        event_time: int,
    ) -> TemporalSceneNarrationExample:
        from sceneops.training.temporal_scene_narration_dataset import (
            TemporalSceneNarrationDatasetGenerator,
        )

        example = TemporalSceneNarrationDatasetGenerator._build_example(
            scenario_id=scenario_id,
            patient_id=patient_id,
            observation_type=observation_type,
            initial_observation=initial_observation,
            follow_up_observation=follow_up_observation,
        )
        return example.model_copy(
            update={
                "example_id": (
                    f"{scenario_id}-{patient_id}-{observation_type.value}-"
                    f"at-{event_time:04d}"
                ),
                "generator_version": TEMPORAL_SCENE_GENERATOR_V3_VERSION,
            }
        )

    @staticmethod
    def deduplicate_exact_examples(
        examples: list[TrainingExample],
    ) -> list[TrainingExample]:
        """Remove exact prompt/response duplicates while preserving order."""
        seen = set()
        unique = []
        for example in examples:
            signature = tuple(
                (message.role, message.content) for message in example.messages
            )
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(example)
        return unique

    @classmethod
    def validate_full_dataset(
        cls,
        examples: list[TrainingExample],
        *,
        expected_scenarios: int | None = None,
    ) -> dict:
        if not examples:
            raise ValueError("Dataset cannot be empty.")

        ids = [example.example_id for example in examples]
        if len(ids) != len(set(ids)):
            raise ValueError("Dataset contains duplicate example IDs.")

        scenario_ids = {example.scenario_id for example in examples}
        if expected_scenarios is not None and len(scenario_ids) != expected_scenarios:
            raise ValueError(
                f"Expected {expected_scenarios} scenarios; found {len(scenario_ids)}."
            )

        scene_examples = [
            item for item in examples if isinstance(item, SceneNarrationExample)
        ]
        temporal_examples = [
            item for item in examples if isinstance(item, TemporalSceneNarrationExample)
        ]
        expected_per_task = len(scenario_ids) * 20
        if len(scene_examples) != expected_per_task:
            raise ValueError(
                f"Expected {expected_per_task} scene examples; found {len(scene_examples)}."
            )
        if len(temporal_examples) != expected_per_task:
            raise ValueError(
                f"Expected {expected_per_task} temporal examples; "
                f"found {len(temporal_examples)}."
            )

        for example in scene_examples:
            if len(example.messages) != 3:
                raise ValueError(f"{example.example_id}: expected three messages.")
            if tuple(message.role for message in example.messages) != (
                "system",
                "user",
                "assistant",
            ):
                raise ValueError(f"{example.example_id}: invalid message roles.")
            serialized = " ".join(message.content for message in example.messages)
            for fact in example.source_facts:
                expected_prefix = f"patient_observation:{example.patient_id}:"
                if not fact.fact.startswith(expected_prefix):
                    raise ValueError(
                        f"{example.example_id}: source fact patient identity mismatch."
                    )
                if "observed at simulation second 600" not in fact.value:
                    raise ValueError(
                        f"{example.example_id}: scene fact timestamp is not final checkpoint."
                    )
                if cls._fact_value(fact) not in serialized:
                    raise ValueError(
                        f"{example.example_id}: source observation omitted from messages."
                    )
            for pattern in TemporalSceneNarrationValidator.FORBIDDEN_PATTERNS:
                if pattern.search(serialized):
                    raise ValueError(
                        f"{example.example_id}: hidden simulator metadata in messages."
                    )

        TemporalSceneNarrationValidator.validate(temporal_examples)

        per_scenario: dict[str, Counter] = {}
        for example in examples:
            per_scenario.setdefault(example.scenario_id, Counter())
            kind = "scene" if isinstance(example, SceneNarrationExample) else "temporal"
            per_scenario[example.scenario_id][kind] += 1
        for scenario_id, counts in per_scenario.items():
            if counts != Counter({"scene": 20, "temporal": 20}):
                raise ValueError(
                    f"{scenario_id}: expected 20 scene and 20 temporal examples; "
                    f"found {dict(counts)}."
                )

        temporal_signatures = {
            (
                example.observation_type.value,
                tuple(cls._fact_value(fact) for fact in example.initial_facts),
                tuple(cls._fact_value(fact) for fact in example.follow_up_facts),
            )
            for example in temporal_examples
        }
        exact_record_signatures = {
            json.dumps(example.to_training_record(), sort_keys=True)
            for example in examples
        }

        return {
            "total_examples": len(examples),
            "unique_scenarios": len(scenario_ids),
            "unique_example_ids": len(set(ids)),
            "scene_narration_examples": len(scene_examples),
            "temporal_narration_examples": len(temporal_examples),
            "distinct_temporal_panels": len(temporal_signatures),
            "unique_training_records": len(exact_record_signatures),
            "duplicate_training_records": len(examples) - len(exact_record_signatures),
        }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_and_export(
    output_dir: Path,
    seeds: tuple[int, ...] = tuple(range(DEFAULT_SCENARIO_COUNT)),
    minimum_examples: int = MINIMUM_DATASET_EXAMPLES,
) -> dict:
    """Generate, validate, deduplicate, scenario-split and publish the dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = TemporalSceneNarrationDatasetV3Generator()
    raw_examples = generator.generate_full_dataset(seeds)
    raw_validation = generator.validate_full_dataset(
        raw_examples,
        expected_scenarios=len(seeds),
    )
    examples = generator.deduplicate_exact_examples(raw_examples)
    if len(examples) < minimum_examples:
        raise ValueError(
            f"Only {len(examples)} unique training examples remain after exact "
            f"deduplication; minimum required is {minimum_examples}. "
            "Increase scenario coverage or improve genuine data diversity."
        )

    splits = split_by_scenario(
        examples,
        train_ratio=0.6,
        validation_ratio=0.2,
        seed=42,
    )

    master_path = output_dir / "sceneops_narration_master.jsonl"
    split_paths = {
        name: output_dir / f"sceneops_narration_{name}.jsonl"
        for name in ("train", "validation", "test")
    }
    manifest_path = output_dir / "sceneops_narration_manifest.json"
    staged_paths = {
        "master": master_path.with_suffix(".jsonl.tmp"),
        **{name: path.with_suffix(".jsonl.tmp") for name, path in split_paths.items()},
        "manifest": manifest_path.with_suffix(".json.tmp"),
    }

    try:
        _write_jsonl(
            staged_paths["master"],
            [example.model_dump(mode="json") for example in examples],
        )
        for name, split_examples in splits.items():
            _write_jsonl(
                staged_paths[name],
                [example.to_training_record() for example in split_examples],
            )

        task_distribution = {
            name: {
                task: sum(1 for item in records if item.task == task)
                for task in ("scene_narration", "temporal_scene_narration")
            }
            for name, records in splits.items()
        }
        manifest = {
            "dataset_name": "sceneops_scene_and_temporal_narration",
            "dataset_version": "1.0.0",
            "generator_version": FULL_DATASET_GENERATOR_VERSION,
            "python_version": sys.version.split()[0],
            "minimum_required_examples": minimum_examples,
            "raw_examples": raw_validation["total_examples"],
            "duplicates_removed": raw_validation["total_examples"] - len(examples),
            "total_examples": len(examples),
            "unique_scenarios": len({item.scenario_id for item in examples}),
            "unique_example_ids": len({item.example_id for item in examples}),
            "scene_narration_examples": sum(
                isinstance(item, SceneNarrationExample) for item in examples
            ),
            "temporal_narration_examples": sum(
                isinstance(item, TemporalSceneNarrationExample) for item in examples
            ),
            "distinct_temporal_panels_before_deduplication": (
                raw_validation["distinct_temporal_panels"]
            ),
            "unique_training_records": len(examples),
            "duplicate_training_records": 0,
            "split_status": "passed",
            "split_ratios_requested": {
                "train": 0.6,
                "validation": 0.2,
                "test": 0.2,
            },
            "split_examples": {name: len(records) for name, records in splits.items()},
            "split_task_distribution": task_distribution,
            "split_scenarios": {
                name: sorted({item.scenario_id for item in records})
                for name, records in splits.items()
            },
            "files": {
                "master": master_path.name,
                **{name: path.name for name, path in split_paths.items()},
            },
            "training_format": "messages",
            "validation_status": "passed",
            "scenario_seeds": list(seeds),
        }
        _write_jsonl(staged_paths["manifest"], [manifest])

        destinations = {
            "master": master_path,
            **split_paths,
            "manifest": manifest_path,
        }
        for name, destination in destinations.items():
            staged_paths[name].replace(destination)

        manifest["sha256"] = {
            name: _sha256(path)
            for name, path in destinations.items()
            if name != "manifest"
        }
        manifest_tmp = manifest_path.with_suffix(".json.tmp")
        try:
            _write_jsonl(manifest_tmp, [manifest])
            manifest_tmp.replace(manifest_path)
        finally:
            if manifest_tmp.exists():
                manifest_tmp.unlink()
    finally:
        for path in staged_paths.values():
            if path.exists():
                path.unlink()

    return manifest


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    output_dir = project_root / "data" / "processed"
    manifest = generate_and_export(output_dir)
    print("SceneOps narration dataset generated and validated.")
    print(f"Raw examples: {manifest['raw_examples']}")
    print(f"Unique examples: {manifest['total_examples']}")
    print(f"Duplicates removed: {manifest['duplicates_removed']}")
    print(f"Scenarios: {manifest['unique_scenarios']}")
    print(f"Scene narration: {manifest['scene_narration_examples']}")
    print(f"Temporal narration: {manifest['temporal_narration_examples']}")
    print(
        f"Distinct temporal panels: {manifest['distinct_temporal_panels_before_deduplication']}"
    )
    print(f"Duplicate training records: {manifest['duplicate_training_records']}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
