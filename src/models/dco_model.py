from typing import Dict, Optional, List
import torch
from src.configs import DecoderConfigs, ModelConfigs
from src.models.base_model import BaseModel

class DCOModel(BaseModel):
    def __init__(
        self,
        model_configs: ModelConfigs,
        decoder_configs: DecoderConfigs,
    ):
        super().__init__(model_configs, decoder_configs)
        
        self.dco_configs = decoder_configs.configs or {}
        # 获取超参数，提供默认值
        self.beta = self.dco_configs.get("beta", 10.0)
        self.tau = self.dco_configs.get("tau", 0.5)
        
        # 获取干预层范围
        start = self.dco_configs.get("intervention_start_layer", 0)
        end = self.dco_configs.get("intervention_end_layer", self.model.config.num_hidden_layers)
        self.intervention_layers = list(range(start, end))

        print(f"[DCO] Initialized with beta={self.beta}, tau={self.tau}, layers={self.intervention_layers}")

    def _get_dco_params(self) -> Dict:
        """打包参数传递给底层 Llama 模型"""
        return {
            "beta": self.beta,
            "tau": self.tau,
            "layers": self.intervention_layers
        }

    def generate(self, inputs, return_attentions: bool = False) -> dict:
        self.model.eval()
        
        # 准备输入
        prompt = inputs["prompted_question"][0]
        if len(inputs["verbalised_instruction"][0]):
            use_system_prompt = True
        else:
            use_system_prompt = False
        
        tokenised_inputs = self._verbalise_input(
            prompt, use_system_prompt=use_system_prompt
        ).to(self.model.device)

        # 获取 DCO 参数
        dco_params = self._get_dco_params()

        with torch.inference_mode():
            # 1. Prefill 阶段 (同时也应用 DCO 以保持一致性)
            input_logits = self.model(
                input_ids=tokenised_inputs[:, :-1], 
                use_cache=True, 
                return_dict=True,
                dco_params=dco_params 
            )
            
            generated_ids = []
            entropies = []  # <--- [新增1] 初始化熵列表
            last_input_token = tokenised_inputs[:, -1]
            past_kv = input_logits.past_key_values
            
            # 2. Decoding 阶段
            for _ in range(self.max_new_tokens):
                last_input_token = last_input_token.view(1, 1)
                
                outputs = self.model(
                    input_ids=last_input_token,
                    past_key_values=past_kv,
                    use_cache=True,
                    attn_mode=self.attn_mode,
                    dco_params=dco_params # 注入参数
                )
                
                # <--- [新增2] 计算并记录熵
                # outputs.logits shape: [batch, 1, vocab_size]
                current_entropy = self._calculate_entropy(outputs.logits[0, -1]) 
                entropies.append(current_entropy.item())
                # -------------------------

                past_kv = outputs.past_key_values
                next_token = outputs.logits[0, -1].argmax()
                generated_ids.append(next_token.item())
                
                last_input_token = next_token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
            
            decoded_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {
            "decoded_text": decoded_text,
            "alphas": entropies, # <--- [新增3] 将 entropies 放入返回字典，key 必须是 "alphas" 以兼容 run.py
            "attentions": {} 
        }

    def lm_score(self, inputs, answer):
        # 适配 TruthfulQA 等需要计算 PPL 的任务
        self.model.eval()
        prompt = inputs["prompted_question"][0]
        if len(inputs["verbalised_instruction"][0]):
            use_system_prompt = True
        else:
            use_system_prompt = False

        dco_params = self._get_dco_params()

        with torch.no_grad():
            if type(prompt) == list:
                input_text = prompt + [answer]
            else:
                input_text = prompt + answer

            input_ids = self._verbalise_input(
                input_text,
                use_system_prompt=use_system_prompt,
                add_generation_prompt=False,
            ).to(self.model.device)
            
            # 计算 Label 部分的 loss
            prefix_ids = self._verbalise_input(
                prompt, use_system_prompt=use_system_prompt
            ).to(self.model.device)
            continue_ids = input_ids[0, prefix_ids.shape[-1] :]

            # 这里的关键是传入 dco_params
            outputs = self.model(input_ids, attn_mode=self.attn_mode, dco_params=dco_params)[0]
            
            logits = outputs[0, prefix_ids.shape[-1] - 1 : -1, :]
            log_probs = logits.log_softmax(dim=-1)
            token_log_probs = log_probs[range(log_probs.shape[0]), continue_ids].sum().item()

        return token_log_probs