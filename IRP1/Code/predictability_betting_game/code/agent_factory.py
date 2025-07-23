from tsdm.agents import (
    AlwaysUpAgent,
    RepeatLastMovementAgent,
    FrequencyBasedMajorityAgent,
    StaticMeanReversionAgent,
    DynamicMeanReversionAgent,
    SGDClassifierAgent,
    DQNAgent,
)

def create_agent(agent_name, params=None):
    if params is None:
        params = {}

    if agent_name == "Always Up Agent":
        return AlwaysUpAgent()
    elif agent_name == "Repeat Last Movement Agent":
        return RepeatLastMovementAgent()
    elif agent_name == "Frequency-Based Majority Agent":
        return FrequencyBasedMajorityAgent()
    elif agent_name == "Static Mean Reversion Agent":
        return StaticMeanReversionAgent()
    elif agent_name == "Dynamic Mean Reversion Agent":
        return DynamicMeanReversionAgent(**params)
    elif agent_name == "SGD Classifier Agent":
        return SGDClassifierAgent(**params)
    elif agent_name == "DQN Agent":
        return DQNAgent(**params)
    else:
        raise ValueError(f"Unknown agent type: {agent_name}")
