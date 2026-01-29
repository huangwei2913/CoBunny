from transformers import AutoTokenizer

model_path = "/mnt/conda_data/microsoft/phi-1_5"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 1. 查看总词表大小
print(f"Vocab Size: {len(tokenizer)}")

# 2. 查看特殊 Token
print(f"Special Tokens: {tokenizer.special_tokens_map}")

# 3. 打印前 20 个普通词汇
vocab = tokenizer.get_vocab()
# 按照 ID 排序显示
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
print("\nFirst 20 tokens:")
for token, token_id in sorted_vocab[:20]:
    print(f"ID {token_id}: {token}")

# 4. 重点检查：你的自定义 Token 是否在里面
print(f"\n<image> ID: {tokenizer.convert_tokens_to_ids('<image>')}")
print(f"### ID: {tokenizer.convert_tokens_to_ids('###')}")

text = "a\nb"
ids = tokenizer.encode(text, add_special_tokens=False)
tokens = tokenizer.convert_ids_to_tokens(ids)
print(ids)
print(tokens)