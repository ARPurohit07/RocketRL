import torch
import torch.nn as nn

from transformers import AutoModel
from peft import LoraConfig, get_peft_model
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def _pythia_backbone():
    backbone = AutoModel.from_pretrained(
        "./models/pythia-70m", local_files_only=True, dtype=torch.float32
    )
    backbone.config.use_cache = False
    return get_peft_model(backbone, LoraConfig(
        r=8, lora_alpha=16, target_modules=["query_key_value"],
        lora_dropout=0.0, bias="none",
    ))


class PythiaController(BaseFeaturesExtractor):

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=128)

        model_path = "./models/pythia-70m"
        if len(observation_space.shape) != 2 or observation_space.shape[0] != 8:
            raise ValueError(
                f"Expected eight stacked states, got {observation_space.shape}"
            )
        state_dim = observation_space.shape[1]
        if state_dim not in (9, 18):
            raise ValueError(f"Unsupported state dimension: {state_dim}")

        self.backbone = _pythia_backbone()

        self.input_adapter = nn.Sequential(
            nn.Linear(state_dim, 512), nn.LayerNorm(512), nn.Tanh()
        )
        self.output_adapter = nn.Sequential(nn.Linear(512, 128), nn.Tanh())
        # The residual path preserves exact low-dimensional state information
        # while the LoRA-adapted transformer learns temporal interactions.
        self.residual = nn.Sequential(nn.Linear(state_dim, 128), nn.LayerNorm(128), nn.Tanh())

    def forward(self, observations):

        # Shape: [batch, 8 timesteps, state variables]
        x = self.input_adapter(observations.float())

        outputs = self.backbone(inputs_embeds=x, use_cache=False, return_dict=True)

        # Latest timestep representation
        hidden = outputs.last_hidden_state[:, -1, :]

        return self.output_adapter(hidden) + self.residual(observations[:, -1, :].float())


class PythiaDualAxis(BaseFeaturesExtractor):
    """Apply one shared Pythia representation to X and Y rocket views."""

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=256)
        if observation_space.shape != (8, 18):
            raise ValueError("Expected eight stacked 18-value 3D states")
        self.backbone = _pythia_backbone()
        self.input_adapter = nn.Sequential(
            nn.Linear(9, 512), nn.LayerNorm(512), nn.Tanh()
        )
        self.output_adapter = nn.Sequential(nn.Linear(512, 128), nn.Tanh())
        self.residual = nn.Sequential(
            nn.Linear(9, 128), nn.LayerNorm(128), nn.Tanh()
        )

    @staticmethod
    def _axis_view(observations, axis):
        states = observations.float()
        if axis == 0:
            displacement, speed = states[..., 0], states[..., 3]
            angle, omega = -states[..., 6], -states[..., 11] * 3.0 / 5.0
            left = torch.maximum(states[..., 15], states[..., 16])
            right = torch.maximum(states[..., 13], states[..., 14])
        else:
            displacement, speed = states[..., 1], states[..., 4]
            angle, omega = -states[..., 7], states[..., 10] * 3.0 / 5.0
            left = torch.maximum(states[..., 14], states[..., 16])
            right = torch.maximum(states[..., 13], states[..., 15])
        return torch.stack((
            displacement * (12.0 / 15.0),
            (states[..., 2] * 26.0 - 1.5) / 30.0,
            speed * (8.0 / 10.0), states[..., 5] * (8.0 / 10.0),
            angle, omega, left, right, states[..., 17],
        ), dim=-1)

    def forward(self, observations):
        x = self._axis_view(observations, 0)
        y = self._axis_view(observations, 1)
        views = torch.cat((x, y), dim=0)
        embedded = self.input_adapter(views)
        hidden = self.backbone(
            inputs_embeds=embedded, use_cache=False, return_dict=True
        ).last_hidden_state[:, -1, :]
        features = self.output_adapter(hidden) + self.residual(views[:, -1, :])
        x_features, y_features = torch.chunk(features, 2, dim=0)
        return torch.cat((x_features, y_features), dim=-1)
