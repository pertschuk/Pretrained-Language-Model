from collections import defaultdict

DATA_DIR = './'

def eval():
  qrels = []
  with open('./qrels.dev.small', 'r') as qrels_file:
    for line in qrels_file:
      qid, cid = line.rstrip().split('\t')
      qrels.append((qid, cid))

  dev_set = defaultdict(list)
  with open('./top1000.dev', 'r') as dev_file:
    for line in dev_file:
      qid, cid, query, candidate = line.rstrip().split('\t')
      label = 1 if (qid, cid) in qrels else 0
      dev_set[query].append((candidate, label))


if __name__ == '__main__':
  eval()