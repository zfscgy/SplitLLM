from typing import List, Callable

import numpy as np
import torch

from chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
from chatglm.tokenization_chatglm import ChatGLMTokenizer

from transformers import AutoTokenizer, AutoModel


class ChatGML6B:
    pretrained_model_path: str = "/home/zf/projects/chatglm-6b"
    device: str = "cuda:1"

    def __init__(self):
        self.tokenizer: ChatGLMTokenizer = ChatGLMTokenizer.from_pretrained(self.pretrained_model_path)
        self.condgen: ChatGLMForConditionalGeneration = \
            ChatGLMForConditionalGeneration.from_pretrained(self.pretrained_model_path).half().to(self.device)

    def chat(self, query: str):
        return self.condgen.chat(self.tokenizer, query)


    """
    Some customized functions for ChatGLM6B
    1.  Adding noise in the hidden states
    2.  Reconstruct the input given the hidden states
    """
    def get_tokenization(self, query: str):
        tokenization = self.tokenizer(query, return_tensors="pt", padding=True).to(self.device)
        return tokenization.input_ids, tokenization.position_ids, tokenization.attention_mask

    def forward_layers(self, hidden_state: torch.Tensor, position_ids: torch.Tensor, attention_masks: torch.Tensor,
                       start_layer: int, end_layer: int):
        for i, layer in enumerate(self.condgen.transformer.layers[start_layer:end_layer]):
            hidden_state = layer(
                hidden_state,
                position_ids=position_ids,
                attention_mask=attention_masks,
                layer_id=torch.tensor(i + start_layer)
            )[0]
        return hidden_state

    def generate_next_token_probs_noisy(self, input_ids, position_ids, attention_masks, split_layer: int = 0, noise_std: float = 0):
        """
        Adding noise in one intermediate layer. By default, no noise is added.
        :param query:
        :param split_layer:
        :param noise_std:
        :return:
        """
        word_embs = self.condgen.transformer.word_embeddings(input_ids)
        hidden_state = word_embs.transpose(0, 1)
        forward_emb = self.forward_layers(hidden_state, position_ids, attention_masks, 0, split_layer)
        # The output of the bottom model

        forward_emb += noise_std * torch.normal(0, 1, forward_emb.shape, device=forward_emb.device)
        # Adding Gaussian noise on the forward embedding

        total_transformer_layers = len(self.condgen.transformer.layers)
        final_emb = self.forward_layers(forward_emb, position_ids, attention_masks, split_layer, total_transformer_layers)

        logits = self.condgen.lm_head(final_emb).permute(1, 0, 2).contiguous()
        # Get the logits on the next position

        probs = torch.softmax(logits, dim=-1)  # [batch, length, n_tokens]
        return probs[0, -1]


    def greedy_generate(self, query: str, prob_generator: Callable, max_gen_length: int=200):
        input_ids, position_ids, attention_masks = self.get_tokenization(query)
        context_size = len(input_ids[0])
        for i in range(max_gen_length):
            probs = prob_generator(input_ids, position_ids, attention_masks)
            next_id = torch.argmax(probs)
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]]).to(self.device)], dim=-1)  # Append the last id
            new_seq_len = len(input_ids[0])
            attention_masks = np.tril(np.ones([1, new_seq_len, new_seq_len]))
            attention_masks[:, :, :context_size] = 1
            attention_masks = (torch.tensor(attention_masks) < 0.5).to(self.device)
            position_ids = torch.cat([
                position_ids,
                torch.tensor([[[context_size - 2], [new_seq_len - context_size + 1]]]).to(self.device)
            ], dim=-1)
            if next_id == self.condgen.generation_config.eos_token_id:
                break

        resp = self.tokenizer.decode(input_ids[0])
        return resp[len(query):]


if __name__ == '__main__':
    model = ChatGML6B()
    print(model.chat("Introduce you, please.")[0])

    def test_greedy_generate():
        print(model.greedy_generate("我操你妈！", model.generate_next_token_probs_noisy))


    test_greedy_generate()
