import pytest

from sceneops.agents.conversation_memory import PatientConversationManager
from sceneops.agents.disclosure import ConversationTurn


def make_turn(
    question: str = "How are you?",
    response: str = "I feel scared.",
) -> ConversationTurn:
    return ConversationTurn(
        user_message=question,
        patient_response=response,
    )


def test_manager_stores_and_returns_conversation():
    manager = PatientConversationManager()

    manager.add_turn("P001", make_turn())

    history = manager.get_history("P001")

    assert len(history) == 1
    assert history[0].user_message == "How are you?"
    assert history[0].patient_response == "I feel scared."


def test_manager_keeps_patient_histories_separate():
    manager = PatientConversationManager()

    manager.add_turn("P001", make_turn())
    manager.add_turn("P002", make_turn("Are you hurt?", "Yes, my leg hurts."))

    assert len(manager.get_history("P001")) == 1
    assert len(manager.get_history("P002")) == 1
    assert manager.get_history("P001")[0].patient_response == "I feel scared."
    assert manager.get_history("P002")[0].patient_response == ("Yes, my leg hurts.")


def test_manager_enforces_maximum_history_limit():
    manager = PatientConversationManager(max_turns=2)

    manager.add_turn("P001", make_turn("Question 1", "Answer 1"))
    manager.add_turn("P001", make_turn("Question 2", "Answer 2"))
    manager.add_turn("P001", make_turn("Question 3", "Answer 3"))

    history = manager.get_history("P001")

    assert len(history) == 2
    assert [turn.user_message for turn in history] == [
        "Question 2",
        "Question 3",
    ]


def test_manager_returns_defensive_copy():
    manager = PatientConversationManager()
    manager.add_turn("P001", make_turn())

    history = manager.get_history("P001")
    history.clear()

    assert len(manager.get_history("P001")) == 1


def test_manager_clears_one_patient_without_affecting_others():
    manager = PatientConversationManager()
    manager.add_turn("P001", make_turn())
    manager.add_turn("P002", make_turn())

    manager.clear_patient("P001")

    assert manager.get_history("P001") == []
    assert len(manager.get_history("P002")) == 1


def test_manager_clears_all_patient_histories():
    manager = PatientConversationManager()
    manager.add_turn("P001", make_turn())
    manager.add_turn("P002", make_turn())

    manager.clear_all()

    assert manager.get_history("P001") == []
    assert manager.get_history("P002") == []


def test_manager_rejects_invalid_max_turns():
    with pytest.raises(ValueError, match="at least 1"):
        PatientConversationManager(max_turns=0)


def test_manager_rejects_empty_patient_id():
    manager = PatientConversationManager()

    with pytest.raises(ValueError, match="cannot be empty"):
        manager.add_turn(" ", make_turn())


def test_manager_rejects_invalid_turn_type():
    manager = PatientConversationManager()

    with pytest.raises(TypeError, match="ConversationTurn"):
        manager.add_turn("P001", "not a conversation turn")


def test_manager_rejects_empty_id_when_reading_or_clearing():
    manager = PatientConversationManager()

    with pytest.raises(ValueError, match="cannot be empty"):
        manager.get_history("")

    with pytest.raises(ValueError, match="cannot be empty"):
        manager.clear_patient(" ")


@pytest.mark.parametrize("invalid_max_turns", [True, False, 1.5, "10", None])
def test_manager_rejects_non_integer_max_turns(invalid_max_turns):
    with pytest.raises(TypeError, match="positive integer"):
        PatientConversationManager(max_turns=invalid_max_turns)


@pytest.mark.parametrize("patient_id", [" P001", "P001 ", " P001 "])
def test_manager_rejects_patient_ids_with_surrounding_whitespace(patient_id):
    manager = PatientConversationManager()

    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        manager.add_turn(patient_id, make_turn())


def test_manager_rejects_non_string_patient_id():
    manager = PatientConversationManager()

    with pytest.raises(TypeError, match="must be a string"):
        manager.get_history(123)
