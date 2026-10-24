"""Read-only authored environment and cached profile rendering for case authors."""

from collections.abc import Mapping
from copy import deepcopy


class EnvironmentView(Mapping):
    def __init__(self, declaration, profiles=(), *, path='environment'):
        self._declaration = deepcopy(dict(declaration))
        self.profiles = tuple(profiles)
        self.path = path
        self._resolved = {}
        self._typed = None

    def __iter__(self):
        return iter(self._declaration)

    def __len__(self):
        return len(self._declaration)

    def __getitem__(self, name):
        if name not in self._declaration:
            raise KeyError(name)
        return deepcopy(self._declaration[name])

    def __getattr__(self, name):
        # Attribute access never silently substitutes a falsy default.
        from scenario.environment_declaration import environment_fields
        if name not in environment_fields():
            raise AttributeError('unknown environment field: ' + name)
        declaration, _ = self._contracts()
        normalized = declaration.to_dict()
        if name in normalized:
            return deepcopy(normalized[name])
        if name in self._declaration:
            return self[name]
        raise AttributeError('environment field is not declared: ' + name)

    def _contracts(self):
        if self._typed is None:
            from scenario.environment_config import read_environment
            self._typed = read_environment(self._declaration, self.path)
        return self._typed

    def __deepcopy__(self, memo):
        return EnvironmentView(self._declaration, self.profiles, path=self.path)

    def render(self, profile):
        if profile not in self.profiles:
            raise ValueError('profile is not selected: ' + str(profile))
        if profile not in self._resolved:
            declaration, resolver = self._contracts()
            self._resolved[profile] = resolver.resolve(declaration, self.path, profile).to_dict()
        return deepcopy(self._resolved[profile])

    def resolved_config(self, profile):
        return self.render(profile)['resolved_config']
