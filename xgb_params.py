xgb_params = {
    'eta': 0.05,
    'max_depth': 16,
    'subsample':0.5,
    'colsample_bytree': 0.6,
    'min_child_weight': 5,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'device':'cuda',
     'tree_method': 'gpu_hist',
    'gamma': 3,
    'lambda': 150,
    'alpha': 150,
    'gpu_id': 0  
}