from __future__ import annotations


class RolloutBuffer:
    def __init__(self):
        self.matrix_states = []
        self.scalar_states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self) -> None:
        self.matrix_states.clear()
        self.scalar_states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def add(self, matrix_state, scalar_state, action, logprob, reward, done, value) -> None:
        self.matrix_states.append(matrix_state)
        self.scalar_states.append(scalar_state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def __len__(self) -> int:
        return len(self.rewards)

