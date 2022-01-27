from enum import Enum


# class DoneType(Enum):
#     RUNNING = 0
#     EPISODE_DONE = 1
#     GAME_END = 2


class ActionType(Enum):
    IDLE = 0
    LEFT = 1
    TOP_LEFT = 2
    TOP = 3
    TOP_RIGHT = 4
    RIGHT = 5
    BOTTOM_RIGHT = 6
    BOTTOM = 7
    BOTTOM_LEFT = 8
    LONG_PASS = 9
    HIGH_PASS = 10
    SHORT_PASS = 11
    SHOT = 12
    SPRINT = 13
    STOP_MOVING = 14
    STOP_SPRINT = 15
    SLIDING = 16
    DRIBBLE = 17
    STOP_DRIBBLE = 18
    

class GameModeType(Enum):
    NORMAL = 0
    KICK_OFF = 1
    GOAL_KICK = 2
    FREE_KICK = 3
    CORNER = 4
    THROW_IN = 5
    PENALTY = 6


class BallPossType(Enum):
    SELF_TEAM = 0
    OPP_TEAM = 1
    FREE = -1


# sticky_index_to_action = [
#     Action.Left,
#     Action.TopLeft,
#     Action.Top,
#     Action.TopRight,
#     Action.Right,
#     Action.BottomRight,
#     Action.Bottom,
#     Action.BottomLeft,
#     Action.Sprint,
#     Action.Dribble
# ]


class PlayerRoleType(Enum):
    GoalKeeper = 0
    CenterBack = 1
    LeftBack = 2
    RightBack = 3
    DefenceMidfield = 4
    CentralMidfield = 5
    LeftMidfield = 6
    RIghtMidfield = 7
    AttackMidfield = 8
    CentralFront = 9
