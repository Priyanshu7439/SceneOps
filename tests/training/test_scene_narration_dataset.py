from sceneops.training.scene_narration_dataset import (
    SceneNarrationDatasetGenerator,
)


def test_generates_scene_narration_examples():
    examples = SceneNarrationDatasetGenerator().generate()

    assert len(examples) == 100


def test_examples_cover_five_scenarios():
    examples = SceneNarrationDatasetGenerator().generate()

    assert len({example.scenario_id for example in examples}) == 5


def test_examples_have_approved_source_facts():
    examples = SceneNarrationDatasetGenerator().generate()

    assert all(example.source_facts for example in examples)

    assert all(
        fact.fact.startswith("patient_observation:")
        for example in examples
        for fact in example.source_facts
    )


def test_training_record_excludes_source_metadata():
    example = SceneNarrationDatasetGenerator().generate()[0]

    record = example.to_training_record()

    assert set(record) == {"messages"}
    assert "source_facts" not in record


def test_examples_have_valid_message_order():
    examples = SceneNarrationDatasetGenerator().generate()

    for example in examples:
        assert [message.role for message in example.messages] == [
            "system",
            "user",
            "assistant",
        ]


def test_generation_is_reproducible():
    generator = SceneNarrationDatasetGenerator()

    first = generator.generate()
    second = generator.generate()

    assert [example.model_dump(mode="json") for example in first] == [
        example.model_dump(mode="json") for example in second
    ]
