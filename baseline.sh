#!/bin/sh


for name in  GCN GIN GraphSage GAT
do
  for dataset in  abide_AAL116 abide_schaefer100
  do
        model="configs/"${dataset}"/TUs_graph_classification_"${name}"_"${dataset}"_100k.json"
        echo ${model}
        python main.py --config  $model --threshold 0.3 --node_feat_transform pearson --gpu_id 0
  done
done
