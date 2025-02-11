# TRAINING SET
# python generate_datasets.py --include_pairwise_pref --skip_test # Train: P
# python generate_datasets.py --include_reward_scoring --skip_test # Train: R

# 1k, 2k, 4k, 8k, 16k, 32k
python generate_datasets.py --include_silver_pairwise --skip_test --max_silver_train 1000 # Train: S1000
python generate_datasets.py --include_silver_pairwise --skip_test --max_silver_train 2000 # Train: S2000
python generate_datasets.py --include_silver_pairwise --skip_test --max_silver_train 4000 # Train: S4000
python generate_datasets.py --include_silver_pairwise --skip_test --max_silver_train 8000 # Train: S8000
# python generate_datasets.py --include_silver_pairwise --skip_test --max_silver_train 16000 # Train: S16000
# python generate_datasets.py --include_silver_pairwise --skip_test --max_silver_train 32000 # Train: S32000
# python generate_datasets.py --include_pairwise_pref --include_silver_pairwise --skip_test # Train: PS


# TESTING SET
python generate_datasets.py --include_reward_scoring --include_pairwise_pref --include_gold_pairwise --include_silver_pairwise --skip_train --include_subedits --include_h_split # Test: RPSGH
