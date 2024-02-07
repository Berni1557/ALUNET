# -*- coding: utf-8 -*-

from enum import Enum

class SaveState(Enum):
    """
    Save state
    """
    SAVE = 1
    LOAD = 2
    CREATE = 3
    NOTHING = 4
