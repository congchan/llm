from transformers import BloomPreTrainedModel, BloomModel, BloomConfig
import torch.nn as nn
import torch


class BloomRewardModel(BloomPreTrainedModel):
    """
    The Bloom Model transformer with a sequence classification head on top (linear layer).
    uses the last token in order to do the classification, as other causal models (e.g. GPT-1) do.
    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """
    config_class = BloomConfig

    def __init__(self, config: BloomConfig):
        super().__init__(config)
        self.num_labels = 1
        self.model = BloomModel(config)
        self.value_head = nn.Linear(config.hidden_size, self.num_labels)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            mc_token_ids=None,
            labels=None,
            lm_labels=None,
            mc_labels=None,
            return_dict=False,
            output_attentions=False,
            output_hidden_states=False,
    ):
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]
        rewards = self.value_head(hidden_states).squeeze(-1)
        pad_token_id = self.config.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.model.config.pad_token_id
        ends = input_ids.shape[1] - (input_ids == pad_token_id).type(torch.int64).sum(dim=1).view(-1, 1)
        ends = torch.clamp(ends - 1, min=0)
        rewards = torch.gather(rewards, 1, ends)
        return rewards