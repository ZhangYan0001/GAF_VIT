#
# import torch
# import torch.nn as nn
# from skopt import BayesSearchCV
# from skopt.space import Real, Integer, Categorical
#
# param_space = {
#   "lr": Real(0.00625, 0.008, prior="log-uniform"),
#   "batch_size": Integer(8, 32),
#   "unfreeze_interval": Integer(30, 60),
#   "drop_rate": Real(0.0, 0.5),
#   "momentum": Real(0.7, 0.99),
#   "weight_decay": Real(1e-5, 1e-3, prior="log-uniform"),
# }
# def optimize_model(params):
#   config = {
#     "datasets_path": "/home/shunlizhang/zy/xj_datasets",
#     "batch_size": params["batch_size"],
#     "device": "cuda" if torch.cuda.is_available() else "cpu",
#     "epochs": 200,
#     "lr": params["lr"],
#     "weight_decay": params["weight_decay"],
#     "momentum": params["momentum"],
#     "feature_dim": 1024,
#     "projection_dim": 1024,
#     "save_model": "./best_model_bayes1.pth",
#     "vit_kwargs": {
#       "img_size": 224,
#       "patch_size": 16,
#       "in_chans": 3,
#       "embed_dim": 1024,
#       "depth": 6,
#       "num_heads": 16,
#       "mlp_ratio": 4,
#       "drop_rate": params["drop_rate"],
#       "attn_drop_rate": 0.0,
#     },
#   }
#   # 加载训练数据
#   train_loader = data_loader(
#     batch_size=config["batch_size"], npy_files=config["datasets_path"]
#   )
#
#   # 初始化模型
#   vit_backbone = vit.ViTBackbone(**config["vit_kwargs"]).to(config["device"])
#   encoder = SimSiamEncoder(
#     base_model=vit_backbone,
#     projection_dim=config["projection_dim"],
#     feature_dim=config["feature_dim"],
#   ).to(config["device"])
#
#   model = SimSiam(encoder).to(config["device"])
#
#   optimizer = optim.SGD(
#     model.parameters(),
#     lr=config["lr"],
#     momentum=config["momentum"],
#     weight_decay=config["weight_decay"],
#   )
#
#   scheduler = optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, T_max=config["epochs"], eta_min=0.001
#   )
#
#   # 训练模型
#   best_loss = float("inf")
#   for epoch in range(config["epochs"]):
#     avg_loss = train_simsiam(model, train_loader, optimizer, config["device"])
#     scheduler.step()
#
#     if avg_loss < best_loss:
#       best_loss = avg_loss
#
#   # 你可以选择返回验证损失、训练损失等作为优化目标
#   return best_loss
#
#
# def bayes_optim():
#   bayes_search = BayesSearchCV(
#     optimize_model,
#     param_space,
#     n_iter=50,
#     random_state=42,
#     verbose=1,
#   )
#   bayes_search.fit(None)
#
#   best_param = bayes_search.best_params_
#   print("Best Hyperparameters found: ", bayes_search.best_params_)
#
#   with open("./best_hyperparameters.json", "w") as f:
#     json.dump(best_param, f, indent=4)