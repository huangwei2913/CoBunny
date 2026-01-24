import torch
import os
import glob

def check_tensor(name, tensor):
    """检查单个张量的数值健康状况"""
    status = "OK"
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()
    
    # FP16 的临界值是 65504，我们检查超过 65500 的值
    extreme_count = (tensor.abs() > 65500).sum().item()
    
    max_val = tensor.max().item()
    min_val = tensor.min().item()
    avg_val = tensor.mean().item()

    if nan_count > 0 or inf_count > 0 or extreme_count > 0:
        status = "❌ DANGER"
    
    return {
        "status": status,
        "nan": nan_count,
        "inf": inf_count,
        "extreme": extreme_count,
        "range": f"[{min_val:.4f}, {max_val:.4f}]",
        "avg": f"{avg_val:.4f}"
    }

def audit_checkpoint(ckpt_path):
    print(f"\n{'='*80}")
    print(f"🔍 正在审计 Checkpoint: {ckpt_path}")
    print(f"{'='*80}")

    # 扫描该目录下所有的 .bin 文件
    bin_files = glob.glob(os.path.join(ckpt_path, "*.bin"))
    
    for bin_file in bin_files:
        file_name = os.path.basename(bin_file)
        print(f"\n📦 文件: {file_name}")
        print(f"{'-'*40}")
        
        try:
            # 使用 map_location='cpu' 避免占用显存
            weights = torch.load(bin_file, map_location='cpu')
            
            # 兼容有些保存格式是 dict 的情况
            if not isinstance(weights, dict):
                print(f"⚠️ 跳过: {file_name} 格式不是字典")
                continue

            for key, tensor in weights.items():
                if not isinstance(tensor, torch.Tensor):
                    continue
                
                res = check_tensor(key, tensor)
                if res["status"] != "OK":
                    print(f"{res['status']} | Key: {key}")
                    print(f"   - NaN: {res['nan']} | Inf: {res['inf']} | >65500: {res['extreme']}")
                    print(f"   - Range: {res['range']} | Avg: {res['avg']}")
                
            print(f"✅ {file_name} 扫描完成")

        except Exception as e:
            print(f"❌ 读取失败: {e}")

if __name__ == "__main__":
    # 指向你刚才 ls 的目录
    target_dir = "/mnt/CoBunny/checkpoints-finetune/bunny-phi1.5-mixed-lora-695k/checkpoint-2000"
    audit_checkpoint(target_dir)