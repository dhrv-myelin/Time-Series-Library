export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name precursor \
  --is_training 1 \
  --model_id NCDE \
  --model NCDE \
  --data MSL \
  --root_path ./dataset/MSL/ \
  --seq_len 100 \
  --label_len 100 \
  --pred_len 0 \
  --enc_in 55 \
  --c_out 1 \
  --d_model 64 \
  --batch_size 128 \
  --train_epochs 1 \
  --des 'smoke_test' \
  --poa_horizon 50 # --pred_len 100 \
