"""Minimal detector registry."""


class Registry:
    def __init__(self):
        self.data = {}

    def register_module(self, module_name=None):
        def register(cls):
            self.data[module_name or cls.__name__] = cls
            return cls
        return register

    def __getitem__(self, key):
        return self.data[key]


DETECTOR = Registry()
