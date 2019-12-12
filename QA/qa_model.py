from transformers import *
import torch
import numpy as np

MODEL = 'distilbert-base-uncased-distilled-squad'
max_query_length = 64
max_seq_length = 512


def _is_whitespace(c):
  if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
    return True
  return False

def run_squad(question, context):
  model = DistilBertForQuestionAnswering.from_pretrained(MODEL)
  tokenizer = DistilBertTokenizer.from_pretrained(MODEL)

  doc_tokens = []
  char_to_word_offset = []
  prev_is_whitespace = True

  # Split on whitespace so that different tokens may be attributed to their original position.
  for c in context:
    if _is_whitespace(c):
      prev_is_whitespace = True
    else:
      if prev_is_whitespace:
        doc_tokens.append(c)
      else:
        doc_tokens[-1] += c
      prev_is_whitespace = False
    char_to_word_offset.append(len(doc_tokens) - 1)

  tok_to_orig_index = []
  orig_to_tok_index = []
  all_doc_tokens = []
  for (i, token) in enumerate(doc_tokens):
    orig_to_tok_index.append(len(all_doc_tokens))
    sub_tokens = tokenizer.tokenize(token)
    for sub_token in sub_tokens:
      tok_to_orig_index.append(i)
      all_doc_tokens.append(sub_token)

  spans = []

  truncated_query = tokenizer.encode(question, add_special_tokens=False, max_length=max_query_length)
  sequence_added_tokens = tokenizer.max_len - tokenizer.max_len_single_sentence
  sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair

  encoded_dict = tokenizer.encode_plus(
    question,
    all_doc_tokens,
    max_length=max_seq_length,
    return_tensors='pt',
    add_special_tokens=True
  )

  model.eval()
  with torch.no_grad():
    start_logits, end_logits = model(input_ids=encoded_dict['input_ids'])
    start_logits = start_logits[0][len(truncated_query):]
    end_logits = end_logits[0][len(truncated_query):]

  start_tok = int(np.argmax(start_logits))
  end_tok = int(np.argmax(end_logits[start_tok+1:])) + start_tok
  import pdb
  pdb.set_trace()
  return ' '.join(doc_tokens[tok_to_orig_index[start_tok]:tok_to_orig_index[end_tok]+1])


def test_squad():
  QUESTION = 'What is the South Lake Union Streetcar?'
  CONTEXT = '''
  The South Lake Union Streetcar is a streetcar route in Seattle, Washington, United States. Traveling 1.3 miles (2.1 km), it connects downtown to the South Lake Union neighborhood on Westlake Avenue, Terry Avenue, and Valley Street.
  '''
  QUESTION = 'How old is Mark Zuckerberg?'
  CONTEXT = '''
    Mark Zuckerberg is 34 years old.
    '''
  print('Question: %s' % QUESTION)
  print('Answer %s' % run_squad(QUESTION, CONTEXT))


if __name__ == '__main__':
  test_squad()

