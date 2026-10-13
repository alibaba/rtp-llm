"""Read-only authored environment and cached profile rendering for case authors."""

from collections.abc import Mapping
from copy import deepcopy


class EnvironmentView(Mapping):
    def __init__(self, declaration, profiles=(), *, path='environment'):
        self._declaration = deepcopy(dict(declaration))
        self.profiles = tuple(profiles)
        self.path = path
        self._resolved = {}

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
        if name in self._declaration:
            return self[name]
        if not self.profiles:
            raise AttributeError('environment field requires a selected profile: ' + name)
        rendered = [self.render(profile) for profile in self.profiles]
        if any(row.get(name) != rendered[0].get(name) for row in rendered):
            raise ValueError('environment field differs by profile: ' + name)
        if name not in rendered[0]:
            raise AttributeError('environment field is not declared: ' + name)
        return deepcopy(rendered[0][name])

    def __deepcopy__(self, memo):
        return EnvironmentView(self._declaration, self.profiles, path=self.path)

    def render(self, profile):
        if profile not in self.profiles:
            raise ValueError('profile is not selected: ' + str(profile))
        if profile not in self._resolved:
            from scenario.environment_config import environment
            self._resolved[profile] = environment(self._declaration, self.path, profile)
        return deepcopy(self._resolved[profile])

    def resolved_config(self, profile):
        return self.render(profile)['resolved_config']
