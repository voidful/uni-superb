# speech2unit

## Task ASR
python speech2unit.py --model hubert_layer9_code500 --ds superb --ds_split asr
python speech2unit.py --model hubert_layer9_code500 --ds superb --ds_split asr --feat_norm
python speech2unit.py --model hubert_layer9_code500 --ds superb --ds_split asr --beamsearch
python speech2unit.py --model hubert_layer9_code500 --ds superb --ds_split asr --feat_norm --beamsearch;
python speech2unit.py --model hubert_layer6_code50 --ds superb --ds_split asr
python speech2unit.py --model hubert_layer6_code100 --ds superb --ds_split asr
python speech2unit.py --model hubert_layer6_code200 --ds superb --ds_split asr
python speech2unit.py --model mhubert_layer11_code1000 --ds superb --ds_split asr

python speech2unit_distil.py --model voidful/hubert-tiny-v2-unit --ds superb --ds_split asr;\
python speech2unit_distil.py --model voidful/hubert-tiny-v2-unit --ds superb --ds_split ks;\
python speech2unit_distil.py --model voidful/hubert-tiny-v2-unit --ds ashraq/esc50 --ds_split train;

python speech2unit_distil.py --model voidful/hubert-tiny-v2-unit-beamnorm --ds superb --ds_split asr;

## Task KS
python speech2unit.py --model hubert_layer9_code500 --ds superb --ds_split ks
python speech2unit.py --model hubert_layer6_code50 --ds superb --ds_split ks
python speech2unit.py --model hubert_layer6_code100 --ds superb --ds_split ks
python speech2unit.py --model hubert_layer6_code200 --ds superb --ds_split ks
python speech2unit.py --model mhubert_layer11_code1000 --ds superb --ds_split ks

## Task Audio
python speech2unit.py --model hubert_layer9_code500 --ds ashraq/esc50 --ds_split train
python speech2unit.py --model hubert_layer6_code50 --ds ashraq/esc50 --ds_split train
python speech2unit.py --model hubert_layer6_code100 --ds ashraq/esc50 --ds_split train
python speech2unit.py --model hubert_layer6_code200 --ds ashraq/esc50 --ds_split train
python speech2unit.py --model mhubert_layer11_code1000 --ds ashraq/esc50 --ds_split train;

python speech2unit.py --model hubert_layer9_code500 --ds ashraq/esc50 --ds_split train --feat_norm;\
python speech2unit.py --model hubert_layer9_code500 --ds ashraq/esc50 --ds_split train --beamsearch;\
python speech2unit.py --model hubert_layer9_code500 --ds ashraq/esc50 --ds_split train --feat_norm --beamsearch;

# Unit2Task

## unit ASR train 
tfkit-train --batch 10 --epoch 20 --savedir ./asr_hubert_layer9_code500 --train ./model_data/superb_asr_train_chunk_30_hubert_layer9_code500.csv  --test ./model_data/superb_asr_validation_chunk_30_hubert_layer9_code500.csv --task seq2seq --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./asr_hubert_layer6_code50 --train ./model_data/superb_asr_train_chunk_30_hubert_layer6_code50.csv  --test ./model_data/superb_asr_validation_chunk_30_hubert_layer6_code50.csv --task seq2seq --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./asr_hubert_layer6_code100 --train ./model_data/superb_asr_train_chunk_30_hubert_layer6_code100.csv  --test ./model_data/superb_asr_validation_chunk_30_hubert_layer6_code100.csv --task seq2seq --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./asr_hubert_layer6_code200 --train ./model_data/superb_asr_train_chunk_30_hubert_layer6_code200.csv  --test ./model_data/superb_asr_validation_chunk_30_hubert_layer6_code200.csv --task seq2seq --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./asr_mhubert_layer11_code1000 --train ./model_data/superb_asr_train_chunk_30_mhubert_layer11_code1000.csv  --test ./model_data/superb_asr_validation_chunk_30_mhubert_layer11_code1000.csv --task seq2seq --config voidful/bart-base-unit

tfkit-train --batch 10 --epoch 20 --savedir ./asr_hubert_layer9_code500_norm_True_beam_False --train ./model_data/superb_asr_train_chunk_30_hubert_layer9_code500_norm_True_beam_False.csv  --test ./model_data/superb_asr_validation_chunk_30_hubert_layer9_code500_norm_True_beam_False.csv --task seq2seq --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./asr_hubert_layer9_code500_norm_True_beam_True --train ./model_data/superb_asr_train_chunk_30_hubert_layer9_code500_norm_True_beam_True.csv  --test ./model_data/superb_asr_validation_chunk_30_hubert_layer9_code500_norm_True_beam_True.csv --task seq2seq --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./asr_hubert_layer9_code500_norm_False_beam_True --train ./model_data/superb_asr_train_chunk_30_hubert_layer9_code500_norm_False_beam_True.csv  --test ./model_data/superb_asr_validation_chunk_30_hubert_layer9_code500_norm_False_beam_True.csv --task seq2seq --config voidful/bart-base-unit

tfkit-train --batch 10 --epoch 20 --savedir ./asr_tinyhubert-v2 --train ./model_data/superb_asr_train_voidful_hubert-tiny-v2-unit.csv  --test ./model_data/superb_asr_test_voidful_hubert-tiny-v2-unit.csv --task seq2seq --config voidful/bart-base-unit

tfkit-train --batch 10 --epoch 30 --savedir ./asr_tinyhubert-v2-normbeam --train ./model_data/superb_asr_train_voidful_hubert-tiny-v2-unit-beamnorm.csv  --test ./model_data/superb_asr_test_voidful_hubert-tiny-v2-unit-beamnorm.csv --task seq2seq --config voidful/bart-base-unit


## unit ASR eval
tfkit-eval --model ./asr_hubert_layer9_code500/ --metric er --valid ./model_data/superb_asr_validation_chunk_30_hubert_layer9_code500.csv
tfkit-eval --model ./asr_hubert_layer9_code500/ --metric er --valid ./model_data/superb_asr_test_chunk_30_hubert_layer9_code500.csv
tfkit-eval --model ./asr_hubert_layer6_code50/ --metric er --valid ./model_data/superb_asr_validation_chunk_30_hubert_layer6_code50.csv
tfkit-eval --model ./asr_hubert_layer6_code50/ --metric er --valid ./model_data/superb_asr_test_chunk_30_hubert_layer6_code50.csv
tfkit-eval --model ./asr_hubert_layer6_code100/ --metric er --valid ./model_data/superb_asr_validation_chunk_30_hubert_layer6_code100.csv
tfkit-eval --model ./asr_hubert_layer6_code100/ --metric er --valid ./model_data/superb_asr_test_chunk_30_hubert_layer6_code100.csv
tfkit-eval --model ./asr_hubert_layer6_code200/ --metric er --valid ./model_data/superb_asr_validation_chunk_30_hubert_layer6_code200.csv
tfkit-eval --model ./asr_hubert_layer6_code200/ --metric er --valid ./model_data/superb_asr_test_chunk_30_hubert_layer6_code200.csv
tfkit-eval --model ./asr_mhubert_layer11_code1000/ --metric er --valid ./model_data/superb_asr_validation_chunk_30_mhubert_layer11_code1000.csv
tfkit-eval --model ./asr_mhubert_layer11_code1000/ --metric er --valid ./model_data/superb_asr_test_chunk_30_mhubert_layer11_code1000.csv

tfkit-eval --model ./asr_hubert_layer9_code500_norm_True_beam_False/ --metric er --valid ./model_data/superb_asr_test_chunk_30_hubert_layer9_code500_norm_True_beam_False.csv

tfkit-eval --model ./asr_tinyhubert-v2/ --metric er --valid ./model_data/superb_asr_test_voidful_hubert-tiny-v2-unit.csv

tfkit-eval --model ./asr_tinyhubert-v2-normbeam/ --metric er --valid ./model_data/superb_asr_test_voidful_hubert-tiny-v2-unit-beamnorm.csv



## unit KS train
tfkit-train --batch 10 --epoch 20 --savedir ./ks_hubert_layer9_code500 --train ./model_data/superb_ks_train_chunk_30_hubert_layer9_code500.csv  --test ./model_data/superb_ks_validation_chunk_30_hubert_layer9_code500.csv --task clas --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./ks_hubert_layer6_code50 --train ./model_data/superb_ks_train_chunk_30_hubert_layer6_code50.csv  --test ./model_data/superb_ks_validation_chunk_30_hubert_layer6_code50.csv --task clas --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./ks_hubert_layer6_code100 --train ./model_data/superb_ks_train_chunk_30_hubert_layer6_code100.csv  --test ./model_data/superb_ks_validation_chunk_30_hubert_layer6_code100.csv --task clas --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./ks_hubert_layer6_code200 --train ./model_data/superb_ks_train_chunk_30_hubert_layer6_code200.csv  --test ./model_data/superb_ks_validation_chunk_30_hubert_layer6_code200.csv --task clas --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./ks_mhubert_layer11_code1000 --train ./model_data/superb_ks_train_chunk_30_mhubert_layer11_code1000.csv  --test ./model_data/superb_ks_validation_chunk_30_mhubert_layer11_code1000.csv --task clas --config voidful/bart-base-unit

tfkit-train --batch 10 --epoch 20 --savedir ./ks_tinyhubert-v2 --train ./model_data/superb_ks_train_voidful_hubert-tiny-v2-unit.csv  --test ./model_data/superb_ks_test_voidful_hubert-tiny-v2-unit.csv --task clas --config voidful/bart-base-unit


## unit KS eval
tfkit-eval --model ./ks_hubert_layer6_code50/ --metric clas --valid ./model_data/superb_ks_test_chunk_30_hubert_layer6_code50.csv; 
tfkit-eval --model ./ks_hubert_layer6_code100/ --metric clas --valid ./model_data/superb_ks_test_chunk_30_hubert_layer6_code100.csv;
tfkit-eval --model ./ks_hubert_layer6_code200/ --metric clas --valid ./model_data/superb_ks_test_chunk_30_hubert_layer6_code200.csv;
tfkit-eval --model ./ks_hubert_layer9_code500/ --metric clas --valid ./model_data/superb_ks_test_chunk_30_hubert_layer9_code500.csv;
tfkit-eval --model ./ks_mhubert_layer11_code1000/ --metric clas --valid ./model_data/superb_ks_test_chunk_30_mhubert_layer11_code1000.csv;

## unit ESC50 train
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_fold_1_tinyhubert-v2 --train ./model_data/ashraq_esc50_train_voidful_hubert-tiny-v2-unit.csv  --test ./model_data/ashraq_esc50_test_voidful_hubert-tiny-v2-unit.csv --task clas --config voidful/bart-base-unit

tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer9_code500_fold_1 --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_train_fold_1.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_valid_fold_1.csv --task clas  --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer9_code500_fold_2 --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_train_fold_2.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_valid_fold_2.csv --task clas  --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer9_code500_fold_3 --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_train_fold_3.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_valid_fold_3.csv --task clas  --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer9_code500_fold_4 --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_train_fold_4.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_valid_fold_4.csv --task clas  --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer9_code500_fold_5 --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_train_fold_5.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_valid_fold_5.csv --task clas  --config voidful/bart-base-unit;

tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer6_code50_fold_1 --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code50_train_fold_1.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code50_valid_fold_1.csv --task clas --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer6_code50_fold_2 --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code50_train_fold_2.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code50_valid_fold_2.csv --task clas --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer6_code50_fold_3 --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code50_train_fold_3.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code50_valid_fold_3.csv --task clas --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer6_code50_fold_4 --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code50_train_fold_4.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code50_valid_fold_4.csv --task clas --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer6_code50_fold_5 --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code50_train_fold_5.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code50_valid_fold_5.csv --task clas --config voidful/bart-base-unit;

tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer6_code100_fold_1 --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code100_train_fold_1.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code100_valid_fold_1.csv --task clas --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer6_code100_fold_2 --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code100_train_fold_2.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code100_valid_fold_2.csv --task clas --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer6_code100_fold_3 --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code100_train_fold_3.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code100_valid_fold_3.csv --task clas --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer6_code100_fold_4 --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code100_train_fold_4.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code100_valid_fold_4.csv --task clas --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer6_code100_fold_5 --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code100_train_fold_5.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code100_valid_fold_5.csv --task clas --config voidful/bart-base-unit

tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer6_code200_fold_1 --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code200_train_fold_1.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code200_valid_fold_1.csv --task clas --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer6_code200_fold_2 --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code200_train_fold_2.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code200_valid_fold_2.csv --task clas --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer6_code200_fold_3 --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code200_train_fold_3.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code200_valid_fold_3.csv --task clas --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer6_code200_fold_4 --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code200_train_fold_4.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code200_valid_fold_4.csv --task clas --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer6_code200_fold_5 --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code200_train_fold_5.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code200_valid_fold_5.csv --task clas --config voidful/bart-base-unit

tfkit-train --batch 10 --epoch 20 --savedir ./esc50_mhubert_layer11_code1000_fold_1 --train ./model_data/ashraq_esc50_train_train_chunk_30_mhubert_layer11_code1000_train_fold_1.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_mhubert_layer11_code1000_valid_fold_1.csv --task clas --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_mhubert_layer11_code1000_fold_2 --train ./model_data/ashraq_esc50_train_train_chunk_30_mhubert_layer11_code1000_train_fold_2.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_mhubert_layer11_code1000_valid_fold_2.csv --task clas --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_mhubert_layer11_code1000_fold_3 --train ./model_data/ashraq_esc50_train_train_chunk_30_mhubert_layer11_code1000_train_fold_3.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_mhubert_layer11_code1000_valid_fold_3.csv --task clas --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_mhubert_layer11_code1000_fold_4 --train ./model_data/ashraq_esc50_train_train_chunk_30_mhubert_layer11_code1000_train_fold_4.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_mhubert_layer11_code1000_valid_fold_4.csv --task clas --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_mhubert_layer11_code1000_fold_5 --train ./model_data/ashraq_esc50_train_train_chunk_30_mhubert_layer11_code1000_train_fold_5.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_mhubert_layer11_code1000_valid_fold_5.csv --task clas --config voidful/bart-base-unit

tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer9_code500_fold_1_norm --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_True_beam_False_train_fold_1.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_True_beam_False_valid_fold_1.csv --task clas  --config voidful/bart-base-unit
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer9_code500_fold_2_norm --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_True_beam_False_train_fold_2.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_True_beam_False_valid_fold_2.csv --task clas  --config voidful/bart-base-unit;\
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer9_code500_fold_3_norm --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_True_beam_False_train_fold_3.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_True_beam_False_valid_fold_3.csv --task clas  --config voidful/bart-base-unit;\
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer9_code500_fold_4_norm --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_True_beam_False_train_fold_4.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_True_beam_False_valid_fold_4.csv --task clas  --config voidful/bart-base-unit;\
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer9_code500_fold_5_norm --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_True_beam_False_train_fold_5.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_True_beam_False_valid_fold_5.csv --task clas  --config voidful/bart-base-unit;

tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer9_code500_fold_1_beam --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_False_beam_True_train_fold_1.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_False_beam_True_valid_fold_1.csv --task clas  --config voidful/bart-base-unit;\
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer9_code500_fold_2_beam --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_False_beam_True_train_fold_2.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_False_beam_True_valid_fold_2.csv --task clas  --config voidful/bart-base-unit;\
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer9_code500_fold_3_beam --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_False_beam_True_train_fold_3.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_False_beam_True_valid_fold_3.csv --task clas  --config voidful/bart-base-unit;\
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer9_code500_fold_4_beam --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_False_beam_True_train_fold_4.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_False_beam_True_valid_fold_4.csv --task clas  --config voidful/bart-base-unit;\
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer9_code500_fold_5_beam --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_False_beam_True_train_fold_5.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_False_beam_True_valid_fold_5.csv --task clas  --config voidful/bart-base-unit;

tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer9_code500_fold_1_norm_beam --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_True_beam_True_train_fold_1.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_True_beam_True_valid_fold_1.csv --task clas  --config voidful/bart-base-unit;\
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer9_code500_fold_2_norm_beam --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_True_beam_True_train_fold_2.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_True_beam_True_valid_fold_2.csv --task clas  --config voidful/bart-base-unit;\
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer9_code500_fold_3_norm_beam --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_True_beam_True_train_fold_3.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_True_beam_True_valid_fold_3.csv --task clas  --config voidful/bart-base-unit;\
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer9_code500_fold_4_norm_beam --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_True_beam_True_train_fold_4.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_True_beam_True_valid_fold_4.csv --task clas  --config voidful/bart-base-unit;\
tfkit-train --batch 10 --epoch 20 --savedir ./esc50_hubert_layer9_code500_fold_5_norm_beam --train ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_True_beam_True_train_fold_5.csv  --test ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_True_beam_True_valid_fold_5.csv --task clas  --config voidful/bart-base-unit;

## unit ESC50 eval
tfkit-eval --model ./esc50_hubert_layer9_code500_fold_1_norm_beam/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_True_beam_True_valid_fold_1.csv;\
tfkit-eval --model ./esc50_hubert_layer9_code500_fold_2_norm_beam/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_True_beam_True_valid_fold_2.csv;\
tfkit-eval --model ./esc50_hubert_layer9_code500_fold_3_norm_beam/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_True_beam_True_valid_fold_3.csv;\
tfkit-eval --model ./esc50_hubert_layer9_code500_fold_4_norm_beam/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_True_beam_True_valid_fold_4.csv;\
tfkit-eval --model ./esc50_hubert_layer9_code500_fold_5_norm_beam/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_True_beam_True_valid_fold_5.csv;\
tfkit-eval --model ./esc50_hubert_layer9_code500_fold_1_beam/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_False_beam_True_valid_fold_1.csv;\
tfkit-eval --model ./esc50_hubert_layer9_code500_fold_2_beam/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_False_beam_True_valid_fold_2.csv;\
tfkit-eval --model ./esc50_hubert_layer9_code500_fold_3_beam/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_False_beam_True_valid_fold_3.csv;\
tfkit-eval --model ./esc50_hubert_layer9_code500_fold_4_beam/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_False_beam_True_valid_fold_4.csv;\
tfkit-eval --model ./esc50_hubert_layer9_code500_fold_5_beam/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_False_beam_True_valid_fold_5.csv;

tfkit-eval --model ./esc50_hubert_layer9_code500_fold_1_norm/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_True_beam_False_valid_fold_1.csv;\
tfkit-eval --model ./esc50_hubert_layer9_code500_fold_2_norm/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_True_beam_False_valid_fold_2.csv;\
tfkit-eval --model ./esc50_hubert_layer9_code500_fold_3_norm/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_True_beam_False_valid_fold_3.csv;\
tfkit-eval --model ./esc50_hubert_layer9_code500_fold_4_norm/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_True_beam_False_valid_fold_4.csv;\
tfkit-eval --model ./esc50_hubert_layer9_code500_fold_5_norm/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_norm_True_beam_False_valid_fold_5.csv;

tfkit-eval --model ./esc50_hubert_layer9_code500_fold_1/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_valid_fold_1.csv
tfkit-eval --model ./esc50_hubert_layer9_code500_fold_2/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_valid_fold_2.csv
tfkit-eval --model ./esc50_hubert_layer9_code500_fold_3/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_valid_fold_3.csv
tfkit-eval --model ./esc50_hubert_layer9_code500_fold_4/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_valid_fold_4.csv
tfkit-eval --model ./esc50_hubert_layer9_code500_fold_5/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer9_code500_valid_fold_5.csv

tfkit-eval --model ./esc50_hubert_layer6_code50_fold_1/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code50_valid_fold_1.csv
tfkit-eval --model ./esc50_hubert_layer6_code50_fold_2/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code50_valid_fold_2.csv
tfkit-eval --model ./esc50_hubert_layer6_code50_fold_3/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code50_valid_fold_3.csv
tfkit-eval --model ./esc50_hubert_layer6_code50_fold_4/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code50_valid_fold_4.csv
tfkit-eval --model ./esc50_hubert_layer6_code50_fold_5/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code50_valid_fold_5.csv

tfkit-eval --model ./esc50_hubert_layer6_code100_fold_1/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code100_valid_fold_1.csv
tfkit-eval --model ./esc50_hubert_layer6_code100_fold_2/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code100_valid_fold_2.csv
tfkit-eval --model ./esc50_hubert_layer6_code100_fold_3/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code100_valid_fold_3.csv
tfkit-eval --model ./esc50_hubert_layer6_code100_fold_4/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code100_valid_fold_4.csv
tfkit-eval --model ./esc50_hubert_layer6_code100_fold_5/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code100_valid_fold_5.csv

tfkit-eval --model ./esc50_hubert_layer6_code200_fold_1/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code200_valid_fold_1.csv
tfkit-eval --model ./esc50_hubert_layer6_code200_fold_2/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code200_valid_fold_2.csv
tfkit-eval --model ./esc50_hubert_layer6_code200_fold_3/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code200_valid_fold_3.csv
tfkit-eval --model ./esc50_hubert_layer6_code200_fold_4/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code200_valid_fold_4.csv
tfkit-eval --model ./esc50_hubert_layer6_code200_fold_5/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_hubert_layer6_code200_valid_fold_5.csv

tfkit-eval --model ./esc50_mhubert_layer11_code1000_fold_1/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_mhubert_layer11_code1000_valid_fold_1.csv
tfkit-eval --model ./esc50_mhubert_layer11_code1000_fold_2/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_mhubert_layer11_code1000_valid_fold_2.csv
tfkit-eval --model ./esc50_mhubert_layer11_code1000_fold_3/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_mhubert_layer11_code1000_valid_fold_3.csv
tfkit-eval --model ./esc50_mhubert_layer11_code1000_fold_4/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_mhubert_layer11_code1000_valid_fold_4.csv
tfkit-eval --model ./esc50_mhubert_layer11_code1000_fold_5/ --metric clas --valid ./model_data/ashraq_esc50_train_train_chunk_30_mhubert_layer11_code1000_valid_fold_5.csv

## Speaker correlation

### prepare data
python create_unit_speaker_correlation.py --ds superb --ds_split ks

## Task SD
python speech2unit.py --model hubert_layer9_code500 --ds superb --ds_split sd
python speech2unit.py --model hubert_layer6_code50 --ds superb --ds_split sd
python speech2unit.py --model hubert_layer6_code100 --ds superb --ds_split sd
python speech2unit.py --model hubert_layer6_code200 --ds superb --ds_split sd
python speech2unit.py --model mhubert_layer11_code1000 --ds superb --ds_split sd



