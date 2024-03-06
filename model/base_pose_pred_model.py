from .base_model import BaseModel
from abc import ABC, abstractmethod

class BasePosePredModel(BaseModel, ABC):
    @abstractmethod
    def set_eval_mode(self):
        pass
    