from typing import Any, List, Callable

import numpy as np
import torch
from torch import optim

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
        self.condgen.requires_grad_(False)

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

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def get_initial_state(self, input_ids: torch.Tensor):
        word_embs = self.condgen.transformer.word_embeddings(input_ids)
        return torch.transpose(word_embs, 0, 1)  # Notice the first dimension is the sequence position!


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
        hidden_state = self.get_initial_state(input_ids)
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

    def reconstruct_from_hidden_sates(self, target_hidden_state: torch.Tensor,
                                      hidden_state_generator: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
                                      position_ids: torch.Tensor = None,
                                      attention_masks: torch.Tensor = None,
                                      stop_cossim: float = 0.99,
                                      max_steps: int = 1000):
        """

        :param target_hidden_state: [seq_len, 1, 4096 (model_dim)]
        :param hidden_state_generator:
        :param split_layer:
        :param position_ids:
        :param attention_masks:
        :param stop_cossim:
        :param max_steps:
        :return:
        """

        seq_len = target_hidden_state.shape[0]
        token_ids, position_ids_auto, attention_masks_auto = self.get_tokenization("Hello" * (seq_len - 2))


        surrogate_initial_state = self.get_initial_state(token_ids)
        surrogate_initial_state = surrogate_initial_state.clone().detach()
        surrogate_initial_state.requires_grad = True

        position_ids = position_ids or position_ids_auto
        attention_masks = attention_masks or attention_masks_auto

        optimizer = optim.SGD([surrogate_initial_state], lr=0.1, momentum=0.9)
        for i in range(max_steps):
            surrogate_hidden_state = hidden_state_generator(surrogate_initial_state, position_ids, attention_masks)
            loss = torch.mean(torch.square(surrogate_hidden_state - target_hidden_state))

            cos_sim = torch.cosine_similarity(surrogate_hidden_state.view(-1), target_hidden_state.view(-1), 0)
            if cos_sim.item() > stop_cossim:
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        word_embeddings = self.condgen.transformer.word_embeddings.weight[:130005]
        # The ChatGLM model has a tokenizer error, the ids after 130005 are invalid hence we drop the logits
        surrogate_word_embedding = surrogate_initial_state[:, 0, :]  # [seq_len, 1, 4096]
        token_ids = []
        for i in range(surrogate_word_embedding.shape[0]):
            cos_sims = torch.cosine_similarity(surrogate_word_embedding[i], word_embeddings, dim=-1)
            token_id = torch.argmax(cos_sims)
            token_ids.append(token_id)

        return token_ids



if __name__ == '__main__':
    model = ChatGML6B()
    print(model.chat("Introduce you, please.")[0])

    def test_greedy_generate():
        print(model.greedy_generate("Hello, how's the weather today?", model.generate_next_token_probs_noisy))

    def test_greedy_generate_with_noise():
        for s in [0.5, 1, 2, 4, 8, 16]:
            def noisy_prob_generate(i, p, a):  # Temporary function
                return model.generate_next_token_probs_noisy(i, p, a, split_layer=10, noise_std=s)

            print(f"Noise_std={s:6.2f}", end="  ")
            print(model.greedy_generate("Hello, how's the weather today?", noisy_prob_generate))

    def test_reconstruction():
        query = "抗日战争爆发后，任中共关中地委书记、专员公署专员、军分区和关中警备区第一旅政委。1942年7月调任中共西北中央局党校校长。" \
                "1943年2月任中共绥德地委书记兼绥(德)米(脂)警备区和独立第一旅政委。" \
                "1945年6月当选为中共第七届中央候补委员。同年7月任陕甘宁边区集团军政委，与司令员王世泰率部在淳化爷台山地区反击-军进犯。" \
                "抗日战争胜利后，曾任中共中央组织部副部长。"

        input_ids, position_ids, attention_mask = model.get_tokenization(query)

        def generate_hidden_state(s, p, a):
            return model.forward_layers(s, p, a, start_layer=0, end_layer=20)

        target_state = generate_hidden_state(model.get_initial_state(input_ids), position_ids, attention_mask)

        guessed_tokens = model.reconstruct_from_hidden_sates(target_state, generate_hidden_state)
        print(model.tokenizer.decode(guessed_tokens))


    # test_greedy_generate()
    # test_greedy_generate_with_noise()
    test_reconstruction()