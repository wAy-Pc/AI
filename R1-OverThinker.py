# Idea From：https://gist.github.com/vgel/8a2497dc45b1ded33287fa7bb6cc1adc#file-r1-py

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import gradio as gr
import torch
import random

checkpoint = "./DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint,
                                             torch_dtype=torch.bfloat16,
                                             device_map="auto")

def predict(message, history, min_thinking_tokens=100):
    tokens = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "<think>\n"},
        ],
        continue_final_message=True,
        return_tensors="pt",
    )
    tokens = tokens.to(model.device)
    n_thinking_tokens = 0
    kv = DynamicCache()
    _, _start_think_token, end_think_token = tokenizer.encode("<think></think>")
    answer = ''

    while True:
        out = model(input_ids=tokens, past_key_values=kv, use_cache=True)
        next_token = torch.multinomial(
            torch.softmax(out.logits[0, -1, :], dim=-1), 1
        ).item()
        kv = out.past_key_values

        if (
            next_token in (end_think_token, model.config.eos_token_id)
            and n_thinking_tokens < min_thinking_tokens
        ):
            replacement = random.choice(["\n不对，让我重新分析一下", "\n让我重新思考一下"])
            answer += replacement
            yield answer
            replacement_tokens = tokenizer.encode(replacement)
            n_thinking_tokens += len(replacement_tokens)
            tokens = torch.tensor([replacement_tokens]).to(tokens.device)
        elif next_token == model.config.eos_token_id:
            break
        else:
            answer += tokenizer.decode([next_token])
            yield answer
            n_thinking_tokens += 1
            tokens = torch.tensor([[next_token]]).to(tokens.device)

demo = gr.ChatInterface(predict, 
                        type="messages")

demo.launch()
