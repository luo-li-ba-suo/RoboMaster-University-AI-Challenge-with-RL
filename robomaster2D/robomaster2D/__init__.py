# import logging
from gym.envs.registration import register

# logger = logging.getLogger(__name__)

register(
    id='Robomaster-v0',
    entry_point='robomaster2D.envs:RMUA_Multi_agent_Env',
)
