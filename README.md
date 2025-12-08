# Pytorch implementation of BERT : Pretraining of Bi-directional Transformers for Language Understanding

BERT implementation for Pretraining LLMs software project at UdS.

> [!Note]
> [Update : 03/12/2025] This repository has achieved v1.0.0. 

---

> [!Note]
> This repository is WIP. Stability will be achieved on v1.0.0.

## Usage

### Pretraining run

```bash
bert --model="FacebookAI/roberta-base" \
--memmap_path=<memmap_path> \
--batch_size=2 \
--block_size=128 \
--d_model=256 \
--d_ffn=512 \
--n_heads=4 \
--n_layer=2 \
--dropout=0.0 \
--vocab_size=50265 \
--lr=5e-5 \
--checkpoint_dir='ckpt' \
--log_file='logs/logfile.log' \
--tracking=True \
--wandb_entity=<username> \
--wandb_project_name=<project_name> \
--model_compile=True \
--device=<accelerator> \
--grad_accumulation_steps=4 \
--precision='fp32'
```

### Finetuning run

```bash
add finetuning code here
```

## Data downloading and preprocessing

```bash
add data downloading & preprocessing script 

```

## Pretraining Results

```bash
add loss curves, perplexity, etc

```

## Dashboard

```bash
add w&b link
```

## Checklist

- [X] Model
- [X] Dataloader
- [X] Data processing
- [X] Pretraining
- [X] Mixed precision support
- [ ] Evaluation

```bibtex
@inproceedings{devlin-etal-2019-bert,
    title = "{BERT}: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    author = "Devlin, Jacob  and
      Chang, Ming-Wei  and
      Lee, Kenton  and
      Toutanova, Kristina",
    editor = "Burstein, Jill  and
      Doran, Christy  and
      Solorio, Thamar",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N19-1423/",
    doi = "10.18653/v1/N19-1423",
    pages = "4171--4186",
    abstract = "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models (Peters et al., 2018a; Radford et al., 2018), BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5 (7.7 point absolute improvement), MultiNLI accuracy to 86.7{\%} (4.6{\%} absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement)."
}
```
