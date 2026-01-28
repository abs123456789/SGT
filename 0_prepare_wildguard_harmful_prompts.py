import os
import argparse
from pathlib import Path

from dotenv import load_dotenv
from datasets import load_dataset

def main():
    # 1) 加载同级目录 .env
    load_dotenv()  # 默认会找当前工作目录的 .env
    # 更稳：指定为“脚本同级目录”的 .env
    # load_dotenv(Path(__file__).with_name(".env"))

    # 2) 命令行参数：输出路径
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/mmlu/wildguard_train_harmful.csv",
        help="CSV 输出路径（默认：./data/mmlu/wildguard_train_harmful.csv）",
    )
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 3) 读取 token（优先 HF_TOKEN，其次 HUGGINGFACE_HUB_TOKEN）
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    print("正在加载 WildGuardTrain...")
    ds = load_dataset(
        "allenai/wildguardmix",
        "wildguardtrain",
        split="train",
        token=token,  # 如果你已 login，也可以不传；传了更稳
    )

    print("正在筛选 harmful 样本...")
    harmful = ds.filter(lambda x: x["prompt_harm_label"] == "harmful")
    print(f"原始数量: {len(ds)}, 筛选后有害样本数量: {len(harmful)}")

    harmful = harmful.select_columns(["prompt"])
    harmful.to_csv(str(output_path))

    print(f"✅ 保存路径: {output_path}")
    print("前5条预览：")
    print(harmful[:5])

if __name__ == "__main__":
    main()