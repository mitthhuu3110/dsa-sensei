from typing import List


class InterviewAgent:
    """Simulates interviewer follow-ups for a given topic."""

    def generate_questions(self, topic: str) -> List[str]:
        return [
            f"What is the time complexity of the optimal solution for {topic}?",
            f"Can you describe edge cases for {topic}?",
            f"How would you test your solution for {topic}?",
        ]
