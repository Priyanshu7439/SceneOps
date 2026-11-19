from sceneops.agents.disclosure import ConversationTurn


class PatientConversationManager:
    """Stores bounded, patient-specific dialogue history."""

    def __init__(self, max_turns: int = 10) -> None:
        if isinstance(max_turns, bool) or not isinstance(max_turns, int):
            raise TypeError("max_turns must be a positive integer.")

        if max_turns < 1:
            raise ValueError("max_turns must be at least 1.")

        self.max_turns = max_turns
        self._conversations: dict[str, list[ConversationTurn]] = {}

    @staticmethod
    def _validate_patient_id(patient_id: str) -> None:
        if not isinstance(patient_id, str):
            raise TypeError("patient_id must be a string.")

        if not patient_id.strip():
            raise ValueError("patient_id cannot be empty.")

        if patient_id != patient_id.strip():
            raise ValueError(
                "patient_id cannot contain leading or trailing whitespace."
            )

    def add_turn(
        self,
        patient_id: str,
        turn: ConversationTurn,
    ) -> None:
        """Append a dialogue turn and retain only the latest turns."""

        self._validate_patient_id(patient_id)

        if not isinstance(turn, ConversationTurn):
            raise TypeError("turn must be a ConversationTurn.")

        history = self._conversations.setdefault(patient_id, [])
        history.append(turn)

        if len(history) > self.max_turns:
            del history[: -self.max_turns]

    def get_history(self, patient_id: str) -> list[ConversationTurn]:
        """Return a copy of the patient's conversation history."""

        self._validate_patient_id(patient_id)
        return list(self._conversations.get(patient_id, []))

    def clear_patient(self, patient_id: str) -> None:
        """Clear conversation history for one patient."""

        self._validate_patient_id(patient_id)
        self._conversations.pop(patient_id, None)

    def clear_all(self) -> None:
        """Clear all stored conversation histories."""

        self._conversations.clear()
