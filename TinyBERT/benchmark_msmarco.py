from collections import defaultdict

def main():
  qrels = defaultdict(list)
  with open('qrels.dev.small.tsv') as qrels_file:
    for line in qrels_file:
      qid, _, cid, _ = line.rstrip().split('\t')
      qrels[qid].append(cid)
  with open('msmarco_queries.tsv', 'w') as msmarco_queries:
    with open('queries.dev.small.tsv') as queries_file:
      for line in queries_file:
        qid, query = line.rstrip().split('\t')
        msmarco_queries.write(query + '\t' + ','.join(qrels[qid]) + '\n')


if __name__ == '__main__':
  main()