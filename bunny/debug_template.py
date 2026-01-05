from conversation import conv_templates, SeparatorStyle

def debug_conversation(version_name):
    print(f"\n{'='*20} 测试模版: {version_name} {'='*20}")
    
    if version_name not in conv_templates:
        print(f"❌ 错误: 模版 {version_name} 不存在！")
        return
        
    conv = conv_templates[version_name].copy()
    
    # 模拟数据输入
    user_input = "Describe this image."
    conv.append_message(conv.roles[0], "<image>\n" + user_input) 
    conv.append_message(conv.roles[1], "This is a photo of a cute bunny.")
    
    prompt = conv.get_prompt()
    
    print("【最终生成的 Prompt 字符串】:")
    print("-" * 60)
    # 用 repr() 打印可以看见换行符 \n 和空格
    print(repr(prompt)) 
    print("-" * 60)
    
    print("【关键特征检查】:")
    # 检查 System Prompt
    if conv.system and conv.system in prompt:
        print(f"✅ System Prompt 存在")
    
    # 检查角色标签
    if conv.roles[0] and conv.roles[0] in prompt:
        print(f"✅ 角色 A 存在: '{conv.roles[0]}'")
    if conv.roles[1] and conv.roles[1] in prompt:
        print(f"✅ 角色 B 存在: '{conv.roles[1]}'")

    # 检查结束符 (修正报错点)
    if conv.sep2 is not None:
        if conv.sep2 in prompt:
            print(f"✅ 结束符 (sep2) 存在: '{conv.sep2}'")
        else:
            print(f"❌ 警告: 未在 Prompt 中发现结束符 '{conv.sep2}'")
    else:
        print("⚠️ 该模版没有定义 sep2 (不适合 Stage 2)")

if __name__ == "__main__":
    # 我们主要看 bunny 模版是否符合预期
    debug_conversation("bunny")