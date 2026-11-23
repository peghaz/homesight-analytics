"""Tiny deterministic Triton model for optional protocol integration tests."""

import hashlib
import importlib
from collections.abc import Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray

pb_utils = importlib.import_module("triton_python_backend_utils")


class TritonPythonModel:
    """Return one content-hash vector per supplied VIDEO or TEXT batch row."""

    def execute(self, requests: Sequence[object]) -> list[object]:
        """Accept one optional modality and produce a deterministic finite matrix."""
        responses = list[object]()
        for request in requests:
            tensor = pb_utils.get_input_tensor_by_name(request, "VIDEO")
            if tensor is None:
                tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            if tensor is None:
                responses.append(
                    pb_utils.InferenceResponse(error=pb_utils.TritonError("Supply VIDEO or TEXT"))
                )
                continue
            array = cast(NDArray[np.generic], tensor.as_numpy())
            rows = list[NDArray[np.float32]]()
            for row in array:
                content: bytes = bytes(row[0]) if array.dtype == object else row.tobytes()
                seed: int = int.from_bytes(hashlib.sha256(content).digest()[:8], "big")
                rows.append(np.random.default_rng(seed).standard_normal(16).astype(np.float32))
            values: NDArray[np.float32] = np.stack(rows)
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("EMBEDDING", values)])
            )
        return responses
