
hyperparameters = {
"ddpg": {
    'batch_size': 128, 
    'buffer_size': 50000,
    'learning_rate': 0.001
},                 
"sac": {
    "batch_size": 128,
    "buffer_size": 1000000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
},
"a2c": {
    'n_steps': 5, 
    'ent_coef': 0.01, 
    'learning_rate': 0.0007
}
}