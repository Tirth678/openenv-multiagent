import random
from typing import List, Dict, Optional

class TaskLibrary:
    """Library of pre-defined tasks and subtasks for orchestration."""
    
    def __init__(self):
        self.tasks = [
            {
                "task_id": "code_review_001",
                "name": "Backend API Code Review",
                "description": "Review a Node.js Express API for security vulnerabilities and performance issues.",
                "difficulty": "medium",
                "subtasks": [
                    {"subtask_id": "s1", "name": "Auth check", "required_skill": 0.7, "complexity": 0.6, "dependencies": []},
                    {"subtask_id": "s2", "name": "SQL injection scan", "required_skill": 0.8, "complexity": 0.7, "dependencies": ["s1"]},
                    {"subtask_id": "s3", "name": "Performance audit", "required_skill": 0.75, "complexity": 0.65, "dependencies": ["s1"]},
                    {"subtask_id": "s4", "name": "Write report", "required_skill": 0.6, "complexity": 0.4, "dependencies": ["s2", "s3"]},
                ]
            },
            {
                "task_id": "research_001",
                "name": "Market Research: AI Trends",
                "description": "Gather and summarize the latest trends in Generative AI for enterprise use.",
                "difficulty": "medium",
                "subtasks": [
                    {"subtask_id": "s1", "name": "Data collection", "required_skill": 0.6, "complexity": 0.5, "dependencies": []},
                    {"subtask_id": "s2", "name": "Trend analysis", "required_skill": 0.8, "complexity": 0.8, "dependencies": ["s1"]},
                    {"subtask_id": "s3", "name": "Summarization", "required_skill": 0.7, "complexity": 0.6, "dependencies": ["s2"]},
                ]
            },
            {
                "task_id": "data_analysis_001",
                "name": "Sales Data Visualization",
                "description": "Analyze quarterly sales data and generate Python code for dashboarding.",
                "difficulty": "hard",
                "subtasks": [
                    {"subtask_id": "s1", "name": "Data cleaning", "required_skill": 0.7, "complexity": 0.6, "dependencies": []},
                    {"subtask_id": "s2", "name": "Statistical analysis", "required_skill": 0.85, "complexity": 0.8, "dependencies": ["s1"]},
                    {"subtask_id": "s3", "name": "Plotting script", "required_skill": 0.8, "complexity": 0.7, "dependencies": ["s2"]},
                ]
            },
            {
                "task_id": "writing_001",
                "name": "Technical Blog Post",
                "description": "Write a 1500-word blog post explaining Quantum Computing to beginners.",
                "difficulty": "easy",
                "subtasks": [
                    {"subtask_id": "s1", "name": "Outline", "required_skill": 0.5, "complexity": 0.4, "dependencies": []},
                    {"subtask_id": "s2", "name": "Drafting", "required_skill": 0.7, "complexity": 0.7, "dependencies": ["s1"]},
                    {"subtask_id": "s3", "name": "Editing", "required_skill": 0.6, "complexity": 0.5, "dependencies": ["s2"]},
                ]
            },
            {
                "task_id": "bug_fix_001",
                "name": "Frontend CSS Bug",
                "description": "Fix a layout issue where the sidebar overlaps the main content on mobile.",
                "difficulty": "easy",
                "subtasks": [
                    {"subtask_id": "s1", "name": "Reproduction", "required_skill": 0.4, "complexity": 0.3, "dependencies": []},
                    {"subtask_id": "s2", "name": "CSS fix", "required_skill": 0.6, "complexity": 0.5, "dependencies": ["s1"]},
                ]
            },
            {
                "task_id": "security_001",
                "name": "Penetration Test Report",
                "description": "Perform a simulated pen-test on a web application and write a detailed vulnerability report.",
                "difficulty": "hard",
                "subtasks": [
                    {"subtask_id": "s1", "name": "Port scanning", "required_skill": 0.7, "complexity": 0.6, "dependencies": []},
                    {"subtask_id": "s2", "name": "Vulnerability scan", "required_skill": 0.85, "complexity": 0.8, "dependencies": ["s1"]},
                    {"subtask_id": "s3", "name": "Exploitation proof", "required_skill": 0.9, "complexity": 0.9, "dependencies": ["s2"]},
                    {"subtask_id": "s4", "name": "Report drafting", "required_skill": 0.7, "complexity": 0.6, "dependencies": ["s3"]},
                ]
            },
            {
                "task_id": "doc_gen_001",
                "name": "API Documentation",
                "description": "Generate OpenAPI/Swagger documentation from a set of Python FastAPI routes.",
                "difficulty": "medium",
                "subtasks": [
                    {"subtask_id": "s1", "name": "Route parsing", "required_skill": 0.6, "complexity": 0.5, "dependencies": []},
                    {"subtask_id": "s2", "name": "Schema extraction", "required_skill": 0.75, "complexity": 0.7, "dependencies": ["s1"]},
                    {"subtask_id": "s3", "name": "Swagger YAML generation", "required_skill": 0.7, "complexity": 0.6, "dependencies": ["s2"]},
                ]
            },
            {
                "task_id": "translation_001",
                "name": "App Localization",
                "description": "Translate a mobile app's string resources from English to Japanese and German.",
                "difficulty": "medium",
                "subtasks": [
                    {"subtask_id": "s1", "name": "String extraction", "required_skill": 0.5, "complexity": 0.4, "dependencies": []},
                    {"subtask_id": "s2", "name": "Translation", "required_skill": 0.8, "complexity": 0.8, "dependencies": ["s1"]},
                    {"subtask_id": "s3", "name": "Formatting", "required_skill": 0.6, "complexity": 0.5, "dependencies": ["s2"]},
                ]
            }
        ]

    def get_random_task(self) -> Dict:
        """Returns a random task from the library."""
        return random.choice(self.tasks)

    def get_task_by_id(self, task_id: str) -> Optional[Dict]:
        """Returns a task by its ID, or None if not found."""
        for task in self.tasks:
            if task["task_id"] == task_id:
                return task
        return None

    def get_all_task_names(self) -> List[str]:
        """Returns a list of all task names."""
        return [task["name"] for task in self.tasks]
