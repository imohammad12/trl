import torch
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F
import transformers
from transformers import top_k_top_p_filtering
# from transformers import PegasusForConditionalGeneration, PegasusTokenizer
# from transformers import PegasusPreTrainedModel, PegasusModel, PegasusConfig
from transformers import BartPretrainedModel, BartModel, BartConfig


class BartWithValueHeadModel(BartPretrainedModel):
    """The BartWithValueHeadModel class implements a Bart language model with a secondary, scalar head."""

    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
        r"embed_positions\.weight",
    ]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        config.num_labels = 1
        self.v_head = nn.Linear(config.hidden_size, config.num_labels)
        self.detach_head = False

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def detach_value_head(self):
        self.v_head.detach_head = True

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            #         token_type_ids=None,
            #         position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            mc_token_ids=None,
            lm_labels=None,
            mc_labels=None,
            decoder_input_ids=None,
    ):

        model_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            #             token_type_ids=token_type_ids,
            #             position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
        )

        hidden_states = model_outputs.last_hidden_state

        lm_logits = self.lm_head(hidden_states)

        if self.detach_head:
            value = self.v_head(hidden_states.detach()).squeeze(-1)

        else:
            value = self.v_head(hidden_states).squeeze(-1)

        outputs = lm_logits, model_outputs[1:], value
        return outputs


def respond_to_batch_bart(model, queries, txt_len=20, top_k=0, top_p=1.0, no_explr=False):
    """Sample text from language model."""
    for i in range(txt_len):
        # Get Logits
        output = model(**queries)
        next_token_logits = output[0][:, -1, :]
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

        # Sample
        if not no_explr:
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

        # No exploration (No sampling)
        else:
            next_token = next_token_logits.argmax(-1)

        queries['decoder_input_ids'] = torch.cat([queries['decoder_input_ids'],
                                                  next_token.unsqueeze(-1)],
                                                 dim=-1)
    return queries['decoder_input_ids'][:, -txt_len:]