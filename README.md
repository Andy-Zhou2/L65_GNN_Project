# Tokenized Graph Transformer for Single-Source Shortest Path

## Command list

Example using Hydra-Joblib parallel tasks:

```
# For MPNN experiments
python -m gnn_single_source_shortest_path.gcn_main -m --config-name config seed=1,2,3,4
# For TokenGT experiments
python -m transformers_graph_learner.graph_learner_main -m --config-name config seed=1,2,3,4

```

See hydra-joblib doc: https://hydra.cc/docs/plugins/joblib_launcher/
