import numpy as np
import torch


def free_parameters(model):
    """
    Return optimizer-controlled parameters.

    Frozen parameters, including bias_hh_l0, are excluded.

    The order returned by model.named_parameters()
    is kept fixed and is therefore the parameter-vector layout.
    """

    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


def num_params(model):
    """
    Number of parameters controlled by the optimizer.
    """

    return sum(
        parameter.numel()
        for _, parameter in free_parameters(model)
    )


def layout(model):
    """
    Return the exact parameter-vector layout.

    Each row contains:

        parameter name
        parameter shape
        start index
        end index
    """

    rows = []

    start = 0

    for name, parameter in free_parameters(model):

        count = parameter.numel()

        rows.append(
            (
                name,
                tuple(parameter.shape),
                start,
                start + count
            )
        )

        start += count

    return rows


def get_vector(model):
    """
    Convert optimizer-controlled model parameters
    into one float64 NumPy vector.

    Returns:
        shape = (1608,)
    """

    with torch.no_grad():

        arrays = [
            parameter.detach()
            .cpu()
            .numpy()
            .astype(np.float64)
            .ravel()
            for _, parameter in free_parameters(model)
        ]

        if not arrays:
            return np.empty(
                0,
                dtype=np.float64
            )

        return np.concatenate(arrays)


def set_vector(model, vector):
    """
    Copy a NumPy parameter vector into the model.

    The optimizer works in float64.

    The PyTorch model remains float32.

    No memory aliasing is allowed.
    """

    vector = np.asarray(
        vector,
        dtype=np.float64
    )

    expected = num_params(model)

    if vector.shape != (expected,):

        raise ValueError(
            f"vector shape {vector.shape} "
            f"!= ({expected},)"
        )

    if not np.all(
        np.isfinite(vector)
    ):

        raise ValueError(
            "vector contains NaN/Inf"
        )

    source = torch.from_numpy(vector)

    position = 0

    with torch.no_grad():

        for _, parameter in free_parameters(model):

            count = parameter.numel()

            values = source[
                position:
                position + count
            ]

            values = values.reshape(
                parameter.shape
            )

            # copy_ performs float64 -> float32
            # conversion because the model is float32.
            parameter.copy_(values)

            position += count

    assert position == expected
