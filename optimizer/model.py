import torch
import torch.nn as nn


class PVLSTM(nn.Module):
    """
    RampSense optimizer-compatible PV LSTM.

    Architecture:
        Input:       6 features
        History:     60 minutes
        Hidden:      16
        Output:      8 future PV values

    The PyTorch LSTM normally contains two bias vectors:
        bias_ih_l0
        bias_hh_l0

    We zero and freeze bias_hh_l0 so that the optimizer
    controls 1608 parameters, matching the single-bias
    convention implied by the original paper.

    This is an engineering decision for our adaptation.
    """

    def __init__(
        self,
        input_size=6,
        hidden_size=16,
        output_size=8
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # --------------------------------------------------
        # LSTM
        # --------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # --------------------------------------------------
        # Output layer
        # --------------------------------------------------

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

        # --------------------------------------------------
        # Single-bias convention
        #
        # PyTorch has:
        #   bias_ih_l0
        #   bias_hh_l0
        #
        # We keep bias_ih_l0 and freeze bias_hh_l0 at zero.
        # --------------------------------------------------

        with torch.no_grad():
            self.lstm.bias_hh_l0.zero_()

        self.lstm.bias_hh_l0.requires_grad_(False)

    def forward(self, x):
        """
        x:
            shape = (batch, 60, 6)

        returns:
            shape = (batch, 8)
        """

        out, _ = self.lstm(x)

        # Use the last hidden state.
        last = out[:, -1, :]

        return self.fc(last)


if __name__ == "__main__":

    print("=" * 60)
    print("RampSense Stage A - Model Test")
    print("=" * 60)

    model = PVLSTM()

    # Total PyTorch parameters, including frozen bias_hh.
    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    # Optimizer-controlled parameters only.
    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print("Model:")
    print(model)

    print()
    print("Total PyTorch parameters :", total_params)
    print("Optimizer parameters     :", trainable_params)

    print()
    print("Expected:")
    print("Total PyTorch parameters = 1672")
    print("Optimizer parameters     = 1608")

    # --------------------------------------------------
    # Forward-pass test
    # --------------------------------------------------

    x = torch.randn(
        4,
        60,
        6
    )

    with torch.inference_mode():
        y = model(x)

    print()
    print("Input shape :", tuple(x.shape))
    print("Output shape:", tuple(y.shape))

    assert x.shape == (4, 60, 6)
    assert y.shape == (4, 8)

    assert total_params == 1672
    assert trainable_params == 1608

    assert not model.lstm.bias_hh_l0.requires_grad

    assert torch.all(
        model.lstm.bias_hh_l0 == 0
    )

    print()
    print("MODEL TEST PASSED")
