

echo 'start training'
python train.py
echo 'done'


echo 'inference start'

python inference.py --resize 1024 --model_name '27_MAnet_effi_b7_aug_fdloss_resize1024_' --path '/opt/ml/input/code/workspace/results_baseline_v3/27_Unet_effi_b7_aug_fdloss_06-19-15:39.pt'
# python inference.py --resize 1024 --model_name '27_MAnet_effi_b7_aug_fdloss_resize1024_' --path ''

# python inference.py --model_name 'deeplabv3_effib7_dfloss76' --path '/opt/ml/input/code/workspace/results_baseline_v3/76_deeplabv3+dice+floss_06-18-09:31.pt'
# python inference.py --model_name 'unet_b7_bicubic' --path '/opt/ml/input/code/workspace/results_baseline_v3/48_Unet_efficientb7_06-17-07:07.pt'
# python inference.py --model_name 'unet_b8_aug' --path '/opt/ml/input/code/workspace/results_baseline_v3/35_Unet_efficientb8_aug_06-18-04:15.pt'
# python inference.py --model_name 'deeplabv3_effib7_dfloss50' --path '/opt/ml/input/code/workspace/results_baseline_v3/50_deeplabv3+dice+floss_06-18-09:31.pt' 
# python inference.py --model_name '60_PSPNet_mit_b5_aug_fdloss' --path '/opt/ml/input/code/workspace/results_baseline_v3/60_PSPNet_mit_b5_aug_fdloss_06-18-15:21.pt'



