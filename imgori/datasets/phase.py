from enum import Enum


class Phase(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
