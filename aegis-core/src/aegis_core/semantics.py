from typing import Literal, Union

TokenModifier = Union[
    Literal["declaration"],
    Literal["definition"],
    Literal["readonly"],
    Literal["static"],
    Literal["deprecated"],
    Literal["abstract"],
    Literal["async"],
    Literal["modification"],
    Literal["documentation"],
    Literal["defaultLibrary"],
]

TokenType = Union[
    Literal["comment"],
    Literal["string"],
    Literal["keyword"],
    Literal["number"],
    Literal["regexp"],
    Literal["operator"],
    Literal["namespace"],
    Literal["type"],
    Literal["struct"],
    Literal["class"],
    Literal["interface"],
    Literal["enum"],
    Literal["typeParameter"],
    Literal["function"],
    Literal["method"],
    Literal["decorator"],
    Literal["macro"],
    Literal["variable"],
    Literal["parameter"],
    Literal["property"],
    Literal["label"],
]
