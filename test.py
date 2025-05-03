from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda"  # the device to load the model onto
path = "D:\OceanGPT-7b-v0.2"
model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
).to(device)  # 显式地将模型加载到 GPU 上
tokenizer = AutoTokenizer.from_pretrained(path)

# 确保 pad_token_id 正确设置
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("请说：")
prompt = input()
# prompt = "Which is the largest ocean in the world?"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt", add_special_tokens=True, padding=True).to(device)

# 显式添加 attention_mask
model_inputs["attention_mask"] = (model_inputs.input_ids != tokenizer.pad_token_id).long()

# 生成文本
generated_ids = model.generate(
    model_inputs.input_ids,
    attention_mask=model_inputs.attention_mask,  # 传递 attention_mask
    max_new_tokens=512
)

# 解码生成的 token
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)