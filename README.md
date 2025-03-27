## Command list

Example using Hydra-Joblib parallel tasks:
```
python -m transformers_graph_learner.graph_learner_main -m --config-name config-debug seed=1,2,3,4
```

hydra-joblib doc: https://hydra.cc/docs/plugins/joblib_launcher/

### Command list for main experiments

```
python -m transformers_graph_learner.graph_learner_main -m --config-name config seed=1 training.lr=1e-5,1e-4,1e-3 +dataset.n_node_range=[16,16]
```
```
python -m transformer
s_graph_learner.graph_learner_main -m --config-name config seed=1,2,3,4 training.lr=1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3 +dataset.n_node_range=[16,16] dataset.node_id_encode=laplacian
```