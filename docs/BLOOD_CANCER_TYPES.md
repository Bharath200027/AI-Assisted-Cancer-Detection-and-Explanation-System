# Blood Cancer Types (Configurable)

The system supports **any number of classes** if your dataset folders match those class names, e.g.:

- normal
- all  (Acute Lymphoblastic Leukemia)
- aml  (Acute Myeloid Leukemia)
- cll  (Chronic Lymphocytic Leukemia)
- cml  (Chronic Myeloid Leukemia)

To enable multi-class:
1) Prepare your dataset as:
`data/processed/<name>/{train,val,test}/{normal,all,aml,cll,cml}/...`
2) Update `configs/app.yaml`:
`class_names: ["normal","all","aml","cll","cml"]`
3) Train with `python -m bloodcancer.train ...` or use incremental retrain.

The UI will automatically show buckets and stats for all configured classes.
