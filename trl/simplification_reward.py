
import transformers
import torch
import numpy as np
from tqdm import tqdm
from transformers import DebertaForSequenceClassification, Trainer, TrainingArguments, DebertaTokenizerFast
from pattern.en import lexeme
from sentence_transformers import SentenceTransformer, util


class SimplificationReward:
    """
    A reward model comprised of two scoring model one for semantic similiary score
    and the other one for simplification scores.
    """

    default_params = {
        "device_reward": 'cuda:2' if torch.cuda.is_available() else 'cpu',
        "reward_coef_unsup": 5,
        "alpha": 1.0,
        "beta": 1.0,

    }

    def __init__(self, model_simplicity, tokenizer_simplicity, model_semantic, **params):
        self.params = self.default_params
        self.params.update(params)
        self.model_simplicity = model_simplicity
        self.tokenizer_simplicity = tokenizer_simplicity
        self.model_semantic = model_semantic

        self.model_simplicity.eval()
        self.model_semantic.eval()

    def simplicity_score(self, sents):
        """ returns simplifcity probability """

        toks = self.tokenizer_simplicity(text=sents, truncation=True, padding=True, max_length=100, return_tensors='pt')

        input_ids = toks['input_ids'].to(self.params["device_reward"])
        attention_mask = toks['attention_mask'].to(self.params["device_reward"])
        token_type_ids = toks['token_type_ids'].to(self.params["device_reward"])

        with torch.no_grad():
            output = self.model_simplicity(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                           output_attentions=True,
                                           return_dict=True)

        del input_ids
        del attention_mask
        del token_type_ids

        out = output.logits.squeeze().softmax(dim=-1)[:, 0]

        del output

        return out

    def semantic_sim(self, sentA, sentB):
        """returns the probability that sentA and sentB have the same meaning"""

        # Compute embedding for both lists
        embeddings1 = self.model_semantic.encode(sentA, convert_to_tensor=True)
        embeddings2 = self.model_semantic.encode(sentB, convert_to_tensor=True)

        # Compute cosine-similarities
        cosine_scores = []
        for emb1, emb2 in zip(embeddings1, embeddings2):
            cosine_scores.append(util.pytorch_cos_sim(emb1, emb2)[0][0].unsqueeze(-1))

        cosine_scores = torch.cat(cosine_scores)

        return cosine_scores

    def cal_reward(self, query, response):
        simplicity_reward = self.simplicity_score(response)
        similarity_reward = self.semantic_sim(query, response)

        final_reward = (similarity_reward ** (self.params["alpha"])) * (simplicity_reward ** (self.params["beta"]))
        # final_reward = (simplicity_reward ** (self.params["beta"]))

        del simplicity_reward
        # del similarity_reward

        torch.cuda.empty_cache()

        return final_reward * self.params['reward_coef_unsup']

    def cal_mean_reward(self, queries, responses, forward_batch_size=10):
        rewards = []

        for i in range(int(len(queries) / forward_batch_size)):
            start_index = i * forward_batch_size
            end_index = (i + 1) * forward_batch_size

            rewards.append(self.cal_reward(queries[start_index:end_index], responses[start_index:end_index]))

        mean_rew = torch.cat(rewards).mean()
        print(torch.cat(rewards))
        print(mean_rew)

        return mean_rew, torch.cat(rewards)
