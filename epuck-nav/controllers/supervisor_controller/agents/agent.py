from abc import abstractmethod


class Agent:
    @abstractmethod
    def act(self, *args):
        pass

    @abstractmethod
    def store_transition(self, *args):
        pass

    @abstractmethod
    def train(self, *args):
        pass
