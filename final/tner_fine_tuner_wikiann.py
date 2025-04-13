from tner import GridSearcher

searcher = GridSearcher(
    checkpoint_dir='./ckpt_tner',
    dataset="tner/wikiann-be",  # either of `dataset` (huggingface dataset) or `local_dataset` (custom dataset) should be given
    model="roberta-large",  # language model to fine-tune
    epoch=10,  # the total epoch (`L` in the figure)
    epoch_partial=5,  # the number of epoch at 1st stage (`M` in the figure)
    n_max_config=3,  # the number of models to pass to 2nd stage (`K` in the figure)
    batch_size=16,
    gradient_accumulation_steps=[4, 8],
    crf=[True, False],
    lr=[1e-4, 1e-5],
    weight_decay=[1e-7],
    random_seed=[42],
    lr_warmup_step_ratio=[0.1],
    max_grad_norm=[10]
)
searcher.train()