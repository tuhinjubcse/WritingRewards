python generate_datasets.py --include_pairwise_pref --skip_test
python generate_datasets.py --include_reward_scoring --skip_test
# python generate_datasets.py --include_gold_pairwise
# python generate_datasets.py --include_silver_pairwise
python generate_datasets.py --include_pairwise_pref --include_reward_scoring --skip_test
# python generate_datasets.py --include_pairwise_pref --include_silver_pairwise
# python generate_datasets.py --include_reward_scoring --include_silver_pairwise
# python generate_datasets.py --include_reward_scoring --include_pairwise_pref --include_gold_pairwise --include_silver_pairwise
python generate_datasets.py --include_reward_scoring --include_pairwise_pref --include_gold_pairwise --include_silver_pairwise --skip_train --include_subedits
