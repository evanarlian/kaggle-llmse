#!/usr/bin/env bash


    # parser.add_argument("--pretrained", type=str, required=True)
    # parser.add_argument("--freeze_layers", type=int)
    # parser.add_argument("--max_tokens", type=int, required=True)
    # parser.add_argument("--knn", type=int, required=True)
    # parser.add_argument("--answer_trick", type=str, choices=["no"], default="no")
    # parser.add_argument("--use_lora", action="store_true")
    # parser.add_argument("--lora_r", type=int, default=8)
    # parser.add_argument("--lora_alpha", type=int, default=16)
    # parser.add_argument("--lora_dropout", type=float, default=0.1)
    # # only allow "no" for answer_trick (for now)
    # parser.add_argument("--science_only", action="store_true")
    # parser.add_argument("--title_trick", action="store_true")
    # parser.add_argument("--quick_run", action="store_true")

# LoRA
python src/train_and_inference.py --pretrained=microsoft/deberta-v3-base --max_tokens=256 --knn=4 --use_lora
python src/train_and_inference.py --pretrained=microsoft/deberta-v3-base --max_tokens=256 --knn=4 --use_lora --science_only
python src/train_and_inference.py --pretrained=microsoft/deberta-v3-base --max_tokens=256 --knn=4 --use_lora --title_trick
python src/train_and_inference.py --pretrained=microsoft/deberta-v3-base --max_tokens=256 --knn=4 --use_lora --science_only --title_trick

# finetune
python src/train_and_inference.py --pretrained=microsoft/deberta-v3-base --freeze_layers=10 --max_tokens=256 --knn=4 
python src/train_and_inference.py --pretrained=microsoft/deberta-v3-base --freeze_layers=10 --max_tokens=256 --knn=4 --science_only
python src/train_and_inference.py --pretrained=microsoft/deberta-v3-base --freeze_layers=10 --max_tokens=256 --knn=4 --title_trick
python src/train_and_inference.py --pretrained=microsoft/deberta-v3-base --freeze_layers=10 --max_tokens=256 --knn=4 --science_only --title_trick
