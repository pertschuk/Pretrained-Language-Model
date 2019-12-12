#!/usr/bin/env bash

export FT_BERT_BASE_DIR=./pt-bert-base-uncased-msmarco/
export GENERAL_TINYBERT_DIR=./2nd_General_TinyBERT_4L_312D/
export TASK_DIR=./data
export TMP_TINYBERT_DIR=./tinybert-msmarco-10
export TASK_NAME=msmarco

mkdir $TMP_TINYBERT_DIR
python3 task_distill.py --teacher_model ${FT_BERT_BASE_DIR} \
                       --student_model ${GENERAL_TINYBERT_DIR} \
                       --data_dir ${TASK_DIR} \
                       --task_name ${TASK_NAME} \
                       --output_dir ${TMP_TINYBERT_DIR} \
                       --max_seq_length 128 \
                       --train_batch_size 8 \
                       --num_train_epochs 10 \
                       --do_lower_case

#export TMP_TINYBERT_DIR=./tinybert-msmarco-ft
export TINYBERT_DIR=./tinybert-msmarco-ft
mkdir TINYBERT_DIR
python3 task_distill.py --pred_distill  \
                       --teacher_model ${FT_BERT_BASE_DIR} \
                       --student_model ${TMP_TINYBERT_DIR} \
                       --data_dir ${TASK_DIR} \
                       --task_name ${TASK_NAME} \
                       --output_dir ${TINYBERT_DIR} \
                       --do_lower_case \
                       --learning_rate 3e-5  \
                       --num_train_epochs  3  \
                       --eval_step 100 \
                       --max_seq_length 128 \
                       --train_batch_size 16

python3 task_distill.py --teacher_model ${FT_BERT_BASE_DIR} \
--student_model ${GENERAL_TINYBERT_DIR} \
--teacher_eval \
--data_dir ${TASK_DIR} \
--task_name ${TASK_NAME} \
--output_dir ${TMP_TINYBERT_DIR}

export GLUE_DIR=./glue_data/
export TASK_NAME=MRPC
python3 run_glue.py   --model_type bert   \
--model_name_or_path bert-base-uncased   \
--task_name $TASK_NAME   --do_train   \
--do_eval   --do_lower_case   \
--data_dir $GLUE_DIR/$TASK_NAME   \
--max_seq_length 128   --per_gpu_train_batch_size 32   \
--learning_rate 2e-5   --num_train_epochs 3.0   \
--output_dir /tmp/$TASK_NAME/