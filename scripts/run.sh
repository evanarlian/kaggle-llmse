#!/usr/bin/env bash

    # # required
    # parser.add_argument("--pretrained", type=str, required=True)
    # parser.add_argument("--max_tokens", type=int, required=True)
    # parser.add_argument("--knn", type=int, required=True)
    # parser.add_argument("--ep", type=float, required=True)
    # parser.add_argument("--bs", type=int, required=True)
    # parser.add_argument("--grad_acc", type=int, required=True)
    # # optionals
    # parser.add_argument("--use_lora", action="store_true")
    # parser.add_argument("--lora_r", type=int, default=8)
    # parser.add_argument("--lora_alpha", type=int, default=16)
    # parser.add_argument("--lora_dropout", type=float, default=0.1)
    # parser.add_argument("--freeze_layers", type=int)
    # parser.add_argument("--science_only", action="store_true")
    # parser.add_argument("--title_trick", action="store_true")
    # # enable this flag for fast run and disable wandb
    # parser.add_argument("--quick_run", action="store_true")
    # # DEPRECATED: only allow "no" for answer_trick
    # parser.add_argument("--answer_trick", type=str, choices=["no"], default="no")


# # LoRA (wandb 12-15)
# python src/finetune_mc.py --pretrained=microsoft/deberta-v3-base --max_tokens=256 --knn=4 --ep=1 --bs=4 --grad_acc=8 --use_lora
# python src/finetune_mc.py --pretrained=microsoft/deberta-v3-base --max_tokens=256 --knn=4 --ep=1 --bs=4 --grad_acc=8 --use_lora --science_only
# python src/finetune_mc.py --pretrained=microsoft/deberta-v3-base --max_tokens=256 --knn=4 --ep=1 --bs=4 --grad_acc=8 --use_lora --title_trick
# python src/finetune_mc.py --pretrained=microsoft/deberta-v3-base --max_tokens=256 --knn=4 --ep=1 --bs=4 --grad_acc=8 --use_lora --science_only --title_trick

# # finetune (wandb 16-19)
# python src/finetune_mc.py --pretrained=microsoft/deberta-v3-base --max_tokens=256 --knn=4 --ep=1 --bs=4 --grad_acc=8 --freeze_layers=10 
# python src/finetune_mc.py --pretrained=microsoft/deberta-v3-base --max_tokens=256 --knn=4 --ep=1 --bs=4 --grad_acc=8 --freeze_layers=10 --science_only
# python src/finetune_mc.py --pretrained=microsoft/deberta-v3-base --max_tokens=256 --knn=4 --ep=1 --bs=4 --grad_acc=8 --freeze_layers=10 --title_trick
# python src/finetune_mc.py --pretrained=microsoft/deberta-v3-base --max_tokens=256 --knn=4 --ep=1 --bs=4 --grad_acc=8 --freeze_layers=10 --science_only --title_trick

# # at this point we know that:
# # * lora is worse
# # * title trick or not is not significant enough
# # * science only is worse

# # how about increasing to 2 epoch in science only, can it match the all subset?
# # also for now just use title trick to lessen the experiment (wandb 20-21)
# # turns out the result of science only is very good given 2 epochs (roughly the same data points as all)
# python src/finetune_mc.py --pretrained=microsoft/deberta-v3-base --max_tokens=256 --knn=4 --ep=2 --bs=4 --grad_acc=8 --use_lora --science_only --title_trick
# python src/finetune_mc.py --pretrained=microsoft/deberta-v3-base --max_tokens=256 --knn=4 --ep=2 --bs=4 --grad_acc=8 --freeze_layers=10 --science_only --title_trick

# # does max token length matter? yse, a bit (wandb 22-23)
# python src/finetune_mc.py --pretrained=microsoft/deberta-v3-base --max_tokens=512 --knn=8 --ep=1 --bs=2 --grad_acc=16 --freeze_layers=10 --title_trick
# python src/finetune_mc.py --pretrained=microsoft/deberta-v3-base --max_tokens=512 --knn=8 --ep=2 --bs=2 --grad_acc=16 --freeze_layers=10 --science_only --title_trick

# # RAG
# python src/finetune_rag.py --pretrained=microsoft/phi-1_5 --max_tokens=512 --knn=12 --ep=1 --bs=1 --grad_acc=32 --use_lora --quick_run
