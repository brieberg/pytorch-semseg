OUT_PATH="/home/bijan/Workspace/Python/pytorch-semseg/runs/icnet_isic18/76660/"
MODEL_PATH="${OUT_PATH}icnet_isic18_best_model.pkl"

python3 /home/bijan/Workspace/Python/pytorch-semseg/test.py --model_path $MODEL_PATH --dataset isic18 --img_path /home/bijan/Workspace/Python/pytorch-semseg/data/ISIC18/ISIC2018_Task1-2_Test_Input/ISIC_0012472.jpg --dcrf --out_path "${OUT_PATH}predictedTest12472.jpg"
python3 /home/bijan/Workspace/Python/pytorch-semseg/test.py --model_path $MODEL_PATH --dataset isic18 --img_path /home/bijan/Workspace/Python/pytorch-semseg/data/ISIC18/ISIC2018_Task1-2_Test_Input/ISIC_0012564.jpg --dcrf --out_path "${OUT_PATH}predictedTest12563.jpg"
python3 /home/bijan/Workspace/Python/pytorch-semseg/test.py --model_path $MODEL_PATH --dataset isic18 --img_path /home/bijan/Workspace/Python/pytorch-semseg/data/ISIC18/ISIC2018_Task1-2_Test_Input/ISIC_0012634.jpg --dcrf --out_path "${OUT_PATH}predictedTest12634.jpg"
