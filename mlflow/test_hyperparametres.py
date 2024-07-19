import itertools
import json

param_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_leaf_nodes": [5, 10, 50],
}

param_list = [
    dict(zip(param_grid.keys(), values))
    for values in itertools.product(*param_grid.values())
]

print(json.dumps(param_list, indent=2))
