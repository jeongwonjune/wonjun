

python LT/train_teacher.py \
--epochs 100 \
-b 128 \
--dataset cifar100 \
--train_method mislas \
--num_classes 100 \
--imb_ratio 0.01 \
--finetune \
-a vit \
#--resume /content/drive/MyDrive/project/LT/test/mislas/mixup/cifar100_0.01_vit_epochs100_lr0.0004_bs32_sgd/moco_ckpt.pth.tar \
#--lr 0.001 \

