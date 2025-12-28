import os
import json
import argparse
from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser()
    # 这里的 experiment 现在可以直接接收文件的完整路径
    parser.add_argument('--experiment',
                        type=str,
                        required=True,
                        help="Path to your .jsonl result file")
    args = parser.parse_args()
    return args

def get_gt(data_path):
    GT = {}
    for category in os.listdir(data_path):
        category_dir = os.path.join(data_path, category)
        if not os.path.isdir(category_dir):
            continue
        if os.path.exists(os.path.join(category_dir, 'images')):
            image_path = os.path.join(category_dir, 'images')
            qa_path = os.path.join(category_dir, 'questions_answers_YN')
        else:
            image_path = qa_path = category_dir
        assert os.path.isdir(image_path), image_path
        assert os.path.isdir(qa_path), qa_path
        for file in os.listdir(qa_path):
            if not file.endswith('.txt'):
                continue
            for line in open(os.path.join(qa_path, file)):
                question, answer = line.strip().split('\t')
                GT[(category, file, question)] = answer
    return GT

if __name__ == "__main__":
    args = get_args()

    # 1. 这里已经按你要求的改成了绝对路径
    GT = get_gt(
        data_path='/mnt/CoBunny/eval/mme/MME_Benchmark_release_version/MME_Benchmark'
    )

    # 2. 获取输入文件路径
    input_file = args.experiment 
    
    # 3. 结果输出目录按照你的要求硬指定
    result_dir = "/mnt/CoBunny/mmeanswers"
    os.makedirs(result_dir, exist_ok=True)

    # --- 核心修复：直接打开传入的文件，不再进行路径拼接 ---
    if not os.path.exists(input_file):
        print(f"❌ 错误：找不到文件 {input_file}")
        exit(1)
        
    answers = [json.loads(line) for line in open(input_file)]
    # ----------------------------------------------

    results = defaultdict(list)
    for answer in answers:
        category = answer['question_id'].split('/')[0]
        file = answer['question_id'].split('/')[-1].split('.')[0] + '.txt'
        question = answer['prompt']
        results[category].append((file, answer['prompt'], answer['text']))

    for category, cate_tups in results.items():
        # 输出到指定的硬路径下
        with open(os.path.join(result_dir, f'{category}.txt'), 'w') as fp:
            for file, prompt, answer in cate_tups:
                if 'Answer the question using a single word or phrase.' in prompt:
                    prompt = prompt.replace('Answer the question using a single word or phrase.', '').strip()
                if 'Answer the question directly with a short sentence or phrase.' in prompt:
                    prompt = prompt.replace('Answer the question directly with a short sentence or phrase.', '').strip()
                if 'Please answer yes or no.' not in prompt:
                    prompt = prompt + ' Please answer yes or no.'
                    if (category, file, prompt) not in GT:
                        prompt = prompt.replace(' Please answer yes or no.', '  Please answer yes or no.')
                
                # 获取标准答案并写入
                try:
                    gt_ans = GT[category, file, prompt]
                    tup = (file, prompt, gt_ans, answer)
                    fp.write('\t'.join(tup) + '\n')
                except KeyError:
                    print(f"⚠️ 警告：在标准答案库中找不到条目: {category}, {file}")

    print(f"✨ 转换完成！结果已存入: {result_dir}")