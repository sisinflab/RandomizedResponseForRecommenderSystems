TEMPLATE = """experiment:
  dataset: {dataset}_gen{generated}_eps{epsilon}
  data_config:
    strategy: fixed
    train_path: ../data/{dataset}/generated/{generated}/train_{generated}_eps{epsilon}.tsv
    test_path: ../data/{dataset}/generated/{generated}/test_{generated}.tsv
  top_k: 10
  evaluation:
    cutoffs: [10]
    simple_metrics: [nDCGRendle2020, Recall, HR, nDCG, Precision, F1, MAP, MAR, ItemCoverage, Gini, SEntropy,EFD, EPC,  PopREO, PopRSP, ACLT, APLT, ARP]
  gpu: 0
  models:
    Random:
      meta:
        verbose: True
        save_recs: False
      seed: 42
    MostPop:
      meta:
        verbose: True
        save_recs: False
    ItemKNN:
      meta:
        save_recs: False
        verbose: True
        hyper_max_evals: 5
        hyper_opt_alg: tpe
      neighbors: [uniform, 100, 1000]
      similarity: [cosine, jaccard, dice, pearson]
      implementation: aiolli
      seed: 42
    EASER:
      meta:
        verbose: True
        save_recs: False
        hyper_max_evals: 5
        hyper_opt_alg: tpe
      l2_norm: [uniform, 10, 10e7]
      seed: 42
    RP3beta:
      meta:
        hyper_max_evals: 5
        hyper_opt_alg: tpe
        verbose: True
        save_recs: False
      neighborhood: [uniform, 5, 1000]
      alpha: [uniform, 0, 2]
      beta: [uniform, 0, 2]
      normalize_similarity: [True, False]
      seed: 42"""

TEMPLATE_NO_NOISE = """experiment:
  dataset: {dataset}_gen{generated}_baseline
  data_config:
    strategy: fixed
    train_path: ../data/{dataset}/generated/{generated}/train_{generated}.tsv
    test_path: ../data/{dataset}/generated/{generated}/test_{generated}.tsv
  top_k: 10
  evaluation:
    cutoffs: [10]
    simple_metrics: [nDCGRendle2020, Recall, HR, nDCG, Precision, F1, MAP, MAR, ItemCoverage, Gini, SEntropy,EFD, EPC,  PopREO, PopRSP, ACLT, APLT, ARP]
  gpu: 0
  models:
    Random:
      meta:
        verbose: True
        save_recs: False
      seed: 42
    MostPop:
      meta:
        verbose: True
        save_recs: False
    ItemKNN:
      meta:
        save_recs: False
        verbose: True
        hyper_max_evals: 5
        hyper_opt_alg: tpe
      neighbors: [uniform, 100, 1000]
      similarity: [cosine, jaccard, dice, pearson]
      implementation: aiolli
      seed: 42
    EASER:
      meta:
        verbose: True
        save_recs: False
        hyper_max_evals: 5
        hyper_opt_alg: tpe
      l2_norm: [uniform, 10, 10e7]
      seed: 42
    RP3beta:
      meta:
        hyper_max_evals: 5
        hyper_opt_alg: tpe
        verbose: True
        save_recs: False
      neighborhood: [uniform, 5, 1000]
      alpha: [uniform, 0, 2]
      beta: [uniform, 0, 2]
      normalize_similarity: [True, False]
      seed: 42"""
