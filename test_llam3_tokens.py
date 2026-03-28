from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/mnt/conda_data/Llama-3.2-1B")
test_tokens = ["<image>", "<img_content>", "<pad>", "<|extra_0|>"]

eos_token_id = tokenizer.eos_token_id
pad_token_id  = tokenizer.pad_token_id 
print(f"eos_token_id......: {eos_token_id}")
print(f"pad_token_id......: {pad_token_id}")
bos_token_id = tokenizer.bos_token_id
print(f"bos_token_id......: {bos_token_id}")

for token in test_tokens:
    token_id = tokenizer.convert_tokens_to_ids(token)
    # 如果返回的是 tokenizer.unk_token_id 或者一个非常大的值，说明是新词
    print(f"Token: {token:15} | ID: {token_id}")

