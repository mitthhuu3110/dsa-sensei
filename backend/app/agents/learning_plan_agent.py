from typing import Dict, List


class LearningPlanAgent:
    """Generates a weekly DSA study plan based on user progress."""

    def weekly_plan(self, level: str = "beginner") -> Dict[str, List[str]]:
        core = ["Arrays", "Linked Lists", "Stacks & Queues", "Hashing", "Sorting", "Two Pointers"]
        if level == "intermediate":
            core += ["Binary Search", "Trees", "Heaps", "Graphs Basics"]
        if level == "advanced":
            core += ["DP", "Advanced Graphs", "Tries", "Greedy vs DP"]
        return {"week1": core[:4], "week2": core[4:8], "week3": core[8:12], "week4": core[12:16]}
