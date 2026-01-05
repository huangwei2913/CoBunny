from conversation import conv_templates, SeparatorStyle

def debug_conversation(version_name):
    print(f"\n{'='*20} 测试模版: {version_name} {'='*20}")
    
    # 1. 加载模版
    if version_name not in conv_templates:
        print(f"❌ 错误: 模版 {version_name} 不存在！")
        return
        
    conv = conv_templates[version_name].copy()
    
    # 2. 模拟数据加载过程 (模拟一条简单的 LLaVA 格式数据)
    # 假设图片在第一句话
    user_input = "Describe this image."
    # 你的数据处理逻辑通常会手动添加 <image> token，或者由模版处理
    # 在 conv_bunny 的逻辑里，如果 sep_style 是 TWO，它通常期望 <image> 已经在内容里了
    conv.append_message(conv.roles[0], "<image>\n" + user_input) 
    conv.append_message(conv.roles[1], "This is a photo of a cute bunny.")
    
    # 3. 获取最终 Prompt
    prompt = conv.get_prompt()
    
    # 4. 可视化打印 (把特殊符号高亮显示)
    print("【最终生成的 Prompt 字符串】:")
    print("-" * 60)
    print(prompt)
    print("-" * 60)
    
    # 5. 深度检查关键点
    print("【关键特征检查】:")
    
    # 检查 System Prompt
    if conv.system in prompt:
        print(f"✅ System Prompt 存在: '{conv.system[:20]}...'")
    else:
        print(f"❌ System Prompt 丢失")

    # 检查分隔符 (Sep)
    # conv_bunny 的 sep 是空格，sep2 是 <|endoftext|>
    if conv.sep2 in prompt:
        print(f"✅ 结束符 (EOS) 存在: '{conv.sep2}' (这对模型停止生成至关重要)")
    else:
        print(f"❌ 警告: 结束符 {conv.sep2} 未在 Prompt 中发现 (除非这是多轮对话的中间)")

    # 检查角色
    if conv.roles[0] in prompt and conv.roles[1] in prompt:
        print(f"✅ 角色标签存在: {conv.roles}")
    else:
        print(f"❌ 角色标签丢失")

if __name__ == "__main__":
    # 你的 Stage 1 用的是这个，对比一下
    debug_conversation("plain")
    
    # 你的 Stage 2 应该用这个
    debug_conversation("bunny")