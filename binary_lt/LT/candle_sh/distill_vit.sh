

python LT/distill_featdistill_vit.py \
--epochs 400 \
-b 128 \
--dataset cifar100 \
--num_classes 100 \
--imb_ratio 0.01 \
--pretrained ./test/mislas/cifar100_0.01_resnet152_epochs100_lr0.0004_bs128_sgd/moco_ckpt.best.pth.tar \
--teacher_model 'resnet152' \
--student_model 'bivit' \
--lr 5e-4
#--resume ./test/mislas/mixup/cifar100_0.01_vit_epochs100_lr0.0004_bs32_sgd/student/feature_distill/balance_distill_enc_fc_ver2/no-classwise_bound/no_time/epochs400_lr0.01_bs32_factor_lr0.001/ckpt.pth.tar \
#-p 10 \

#./test/mislas/cifar100_0.01_resnet152_epochs100_lr0.0004_bs128_sgd/moco_ckpt.best.pth.tar \
#./test/mislas/mixup/cifar100_0.01_vit_epochs100_lr0.0004_bs128_sgd/moco_ckpt.best.pth.tar \