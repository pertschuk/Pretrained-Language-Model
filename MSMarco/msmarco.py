import argparse
from tqdm import tqdm, trange
import torch
from transformers import *
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import pathlib


def load_and_cache_triples(triples_path: pathlib.Path, tokenizer):
  cache_path = triples_path.with_suffix('.bin')

  if not cache_path.exists():
    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_labels = []

    with triples_path.open('r') as f:
      for i, line in enumerate(f):
        query, relevant_example, negative_example = line.rstrip().split('\t')

        for passage in (relevant_example, negative_example):
          features = tokenizer.encode_plus(
            query,
            passage,
            add_special_tokens=True,
            max_length=args.max_length,
            return_tensors='pt'
          )
          all_input_ids.append(features['input_ids'])
          all_attention_mask.append(features['attention_mask'])
          all_token_type_ids.append(features['token_type_ids'])
          if i * 2 > args.batch_size * args.steps:
            break
        all_labels.extend([1, 0])

    dataset = TensorDataset(
      torch.tensor(all_input_ids, dtype=torch.long),
      torch.tensor(all_attention_mask, dtype=torch.long),
      torch.tensor(all_token_type_ids, dtype=torch.long),
      torch.tensor(all_labels, dtype=torch.float)
    )
    torch.save(dataset, str(cache_path))

  else:
    dataset = torch.load(str(cache_path))

  return dataset


def train():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  config = BertConfig.from_pretrained(args.model)
  config.num_labels = 1 # regression
  model = BertForSequenceClassification.from_pretrained(args.model, config=config)
  tokenizer = BertTokenizer.from_pretrained(args.model)

  no_decay = ['bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': args.weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}
  ]
  optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
  scheduler = get_linear_schedule_with_warmup(optimizer,
                                              num_warmup_steps=args.warmup_steps,
                                              num_training_steps=args.steps)
  if args.fp16:
    try:
      from apex import amp
    except ImportError:
      raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

  train_dataset = load_and_cache_triples(args.triples_path, tokenizer)
  train_sampler = RandomSampler(train_dataset)
  train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

  global_step = 0
  tr_loss, logging_loss = 0.0, 0.0
  model.zero_grad()
  train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
  for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration")
    for step, batch in enumerate(epoch_iterator):
      model.train()
      batch = tuple(t.to(device) for t in batch)
      inputs = {'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2], # change for distilbert
                'labels': batch[3]}
      outputs = model(**inputs)
      loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

      if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps

      if args.fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
          scaled_loss.backward()
      else:
        loss.backward()

      tr_loss += loss.item()
      if (step + 1) % args.gradient_accumulation_steps == 0:
        if args.fp16:
          torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
          torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        model.zero_grad()
        global_step += 1
        epoch_iterator.set_description("Loss: %s" % (tr_loss/step))
      if (step + 1) % args.save_steps == 0:
        model.save_pretrained('./model.bin')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--triple_train', default='triples.train.small.tsv')
  parser.add_argument('--steps', default=100000)
  parser.add_argument('--save_steps', default=1000)
  parser.add_argument('--model', default='bert-base-uncased')
  parser.add_argument('--batch_size', default=8)
  parser.add_argument('--gradient_accumulation_steps', default=1)
  args = parser.parse_args()
  train()