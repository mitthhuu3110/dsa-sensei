from typing import Dict


class TutorAgent:
    """Explains problems, gives hints, step-by-step breakdown."""

    def plan(self, question: str) -> Dict[str, str]:
        return {
            "goal": "Explain the concept with intuition, steps, and examples.",
            "question": question,
        }
