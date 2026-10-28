import torch
import torch.nn as nn

from transformers import AutoModel
from peft import LoraConfig, get_peft_model
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


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

        backbone = AutoModel.from_pretrained(
            model_path, local_files_only=True, dtype=torch.float32
        )

        backbone.config.use_cache = False

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query_key_value"],
            lora_dropout=0.0,
            bias="none",
        )

        self.backbone = get_peft_model(backbone, lora_config)

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
