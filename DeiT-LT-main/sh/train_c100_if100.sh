 python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port=47773 main.py \
    --model deit_base_distilled_patch16_224 \
    --batch-size 32 \
    --epochs 1200 \
    --gpu 0 \
    --teacher-path teacher/vit_moco_ckpt.best.pth.tar \
    --distillation-type hard \
    --data-path cifar100 \
    --data-set CIFAR100LT \
    --imb_factor 0.01 \
    --output_dir deit_out_c100lt \
    --student-transform 0 \
    --teacher-transform 0 \
    --teacher-model vit \
    --teacher-size 224 \
    --experiment [deitlt_paco_sam_cifar10_if100] \
    --drw 1100 \
    --no-mixup-drw \
    --custom_model \
    --accum-iter 4 \
    --save_freq 300 \
    --weighted-distillation \
    --moco-t 0.05 --moco-k 1024 --moco-dim 32 --feat_dim 64 --paco \
    --repeated-aug \
    --drop-last \
    --bce-loss \
    #--resume deit_out_c100lt/deit_base_distilled_patch16_224_vit_1200_CIFAR100LT_imb100_32_[deitlt_paco_sam_cifar10_if100]/deit_base_distilled_patch16_224_vit_1200_CIFAR100LT_imb100_32_[deitlt_paco_sam_cifar10_if100]_checkpoint.pth
    
    # --bce-loss \
    #--ThreeAugment \

    # --log-results \DeiT-LT-main/moco_ckpt.best.pth.tar
    # --resume deit_out_c100lt/deit_base_distilled_patch16_224_resnet152_800_CIFAR100LT_imb100_32_[deitlt_paco_sam_cifar10_if100]/deit_base_distilled_patch16_224_resnet152_800_CIFAR100LT_imb100_32_[deitlt_paco_sam_cifar10_if100]_checkpoint.pth
   # --resume deit_out_c100lt/deit_base_distilled_patch16_224_vit_800_CIFAR100LT_imb100_32_[deitlt_paco_sam_cifar10_if100]/deit_base_distilled_patch16_224_vit_800_CIFAR100LT_imb100_32_[deitlt_paco_sam_cifar10_if100]_checkpoint.pth