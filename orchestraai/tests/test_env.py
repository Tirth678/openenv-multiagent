import pytest
import numpy as np
import os
import yaml
from env.manager_worker_env import ManagerWorkerEnv
from env.task_library import TaskLibrary
from env.quality_eval import QualityEvaluator
from stable_baselines3.common.env_checker import check_env

@pytest.fixture
def env():
    # Ensure config exists for tests
    if not os.path.exists("config.yaml"):
        # Create a minimal config for testing if needed
        pass
    return ManagerWorkerEnv("config.yaml")

def test_env_reset(env):
    obs, info = env.reset()
    assert isinstance(obs, dict)
    assert "worker_states" in obs
    assert obs["worker_states"].shape == (4, 5)
    assert obs["budget_info"][0] == 1.0 # Initial budget normalized

def test_valid_actions(env):
    env.reset()
    # Test ASSIGN
    obs, reward, term, trunc, info = env.step([0, 0, 0])
    assert info["action_taken"] == "ASSIGN"
    assert env.workers[0]['is_busy'] == True
    
    # Test WAIT
    obs, reward, term, trunc, info = env.step([5, 0, 0])
    assert info["action_taken"] == "WAIT"

def test_budget_depletion(env):
    env.reset()
    env.budget_remaining = 10 # Set low
    # Force expensive actions
    obs, reward, term, trunc, info = env.step([0, 0, 0])
    obs, reward, term, trunc, info = env.step([2, 0, 0]) # REASSIGN
    assert term == True # Should hit 0 budget or steps

def test_completion_bonus(env):
    env.reset()
    # Manually complete all subtasks
    for sid in env.subtask_status:
        env.subtask_status[sid] = "COMPLETED"
    
    # Trigger final step check
    obs, reward, term, trunc, info = env.step([5, 0, 0])
    assert reward >= 100.0 # Completion bonus
    assert term == True

def test_observation_bounds(env):
    obs, info = env.reset()
    for key, val in obs.items():
        if key == "last_action_results":
            assert np.all(val >= -1.0) and np.all(val <= 1.0)
        else:
            assert np.all(val >= 0.0) and np.all(val <= 1.0)

def test_task_library_loading():
    lib = TaskLibrary()
    assert len(lib.tasks) >= 8
    task = lib.get_random_task()
    assert "subtasks" in task

def test_quality_evaluator():
    evaluator = QualityEvaluator()
    subtask = {"complexity": 0.5}
    res = evaluator.evaluate("test", subtask, 0.8)
    assert 0.0 <= res["quality_score"] <= 1.0
    assert isinstance(res["passed"], bool)

def test_env_gymnasium_compliance(env):
    check_env(env)
