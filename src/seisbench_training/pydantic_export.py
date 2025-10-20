from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt


class SeisbenchModelArgs(BaseModel):
    component_order: Literal["ZNE", "ENZ", "Z"] = Field(
        default="ZNE",
        description="Order of the components",
    )
    norm: Literal["peak", "std"] = Field(
        default="peak",
        description="Normalization method",
    )
    phases: Literal["PSN", "NPS"] = Field(
        default="PSN",
        description="Phases to be predicted",
    )
    grouping: Literal["instrument", None] = None
    sampling_rate: PositiveInt = 100
    in_channel: PositiveInt = 3
    classes: PositiveInt = 3

    filter_args: list | dict | None = None
    filter_kwargs: dict | None = None


class ModelDefaultArgs(BaseModel):
    detection_threshold: PositiveFloat = 0.3
    P_threshold: PositiveFloat = 0.3
    S_threshold: PositiveFloat = 0.3

    overlap: PositiveInt | None = None
    blinding: list[PositiveInt] = [250, 250]


class SeisbenchModel(BaseModel):
    model: str
    docstring: str
    model_args: SeisbenchModelArgs
    default_args: ModelDefaultArgs

    seisbench_requirement: str
    version: str

    def get_torch_model(self):
        import seisbench

        model_class = getattr(seisbench.models, self.model)
        model = model_class(**self.model_args.model_dump())
        return model


model_example = SeisbenchModel(
    model="PhaseNet",
    docstring="test",
    model_args=SeisbenchModelArgs(
        component_order="ZNE",
        norm="peak",
    ),
    seisbench_requirement="0.3.0",
    default_args=ModelDefaultArgs(
        detection_threshold=0.5,
        P_threshold=0.6,
        S_threshold=0.4,
    ),
    version="1",
)

json = model_example.model_dump_json(indent=4)
# print(json)


mod = SeisbenchModel.model_validate_json(json)
# print(mod)


path = Path("/home/marius/.seisbench/models/v3/phasenet")

for file in path.glob("*.json*"):
    print(file)
    mod = SeisbenchModel.model_validate_json(file.read_text())
    print(mod.model_dump_json(indent=4))
