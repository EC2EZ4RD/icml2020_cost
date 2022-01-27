from .bandit_teacher import DiscreteTaskBanditTeacher, SimpleTaskBanditTeacher, \
    TaskwiseBanditTeacher, EvalOn5v5BanditTeacher, ContextualBanditTeacher
from .time_teacher import TimeTeacher
from .vacl import VACLTeacher, VACLEPTeacher
from .callbacks import TeacherCallback, TaskwiseTeacherCallback, VACLEPMPETeacherCallback, ContextualTeacherCallback, \
    TimeTeacherCallback, VACLMPETeacherCallback
from .mpe_bandit_teacher import MPEContextualBanditTeacher

TEACHER_LIST = {
    "discrete_bandit": DiscreteTaskBanditTeacher,
    "simple_bandit": SimpleTaskBanditTeacher,
    "task_wise_bandit": TaskwiseBanditTeacher,
    "eval_on_5v5_bandit": EvalOn5v5BanditTeacher,
    "contextual_bandit": ContextualBanditTeacher,
    "mpe_contextual_bandit": MPEContextualBanditTeacher,
    "time_teacher": TimeTeacher,
    "vacl_mpe": VACLTeacher,
    "vacl_ep_mpe": VACLEPTeacher,
}

CALLBACK_LIST = {
    "discrete_bandit": TeacherCallback,
    "simple_bandit": TeacherCallback,
    "task_wise_bandit": TaskwiseTeacherCallback,
    "eval_on_5v5_bandit": TeacherCallback,
    "contextual_bandit": ContextualTeacherCallback,
    "mpe_contextual_bandit": ContextualTeacherCallback,
    "time_teacher": TimeTeacherCallback,
    "vacl_mpe": VACLMPETeacherCallback,
    "vacl_ep_mpe": VACLEPMPETeacherCallback,
}
