from pydantic import BaseModel, ConfigDict

# disable logging protected namespacex
class PyDanticModelBase(BaseModel):
    model_config = ConfigDict(
            protected_namespaces=(),
            arbitrary_types_allowed=True
    )
