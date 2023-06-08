# Preliminary code for Adaptively Sparse Attention

To install the requirements, run:

```python
pip install -r requirements.txt
```

To finetune pretrained model, run:

```python
python train.py --log_dir gpt2-sparse --batch_size 6 --entmax_weight_scheduler cosine_1_8_25000 --sparse_attention true --pretrained gpt2 --int_n_embd 64
```

This will generate a folder `gpt2-sparse` with tensorboard logs under the `logs` folder. Try `python train.py --help` for more options.

To generate text from a finetuned model, run:

```python
python generate.py --model_path <folder-under-logs> --checkpoint <spcify-checkpoint>
```

This repo was heavily inspired by the following

- https://github.com/karpathy/nanoGPT
- https://github.com/deep-spin/entmax/
