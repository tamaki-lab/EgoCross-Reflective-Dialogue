1st EgoCross Challenge @EgoVis, CVPR26
EgoCross is a cross-domain benchmark for evaluating Multimodal Large Language Models (MLLMs) on egocentric video question answering tasks. The benchmark covers four diverse domains with first-person perspective videos. EgoCross website for more information on the EgoCross challenge.

Task Description: Given an egocentric video from a novel domain that differs significantly from commonly seen scenarios (e.g., industrial or surgical environments rather than daily-life settings), the goal is to select the correct answer from four options (A, B, C, D) for a given query question.

We set two tracks:

source-limited track: Participants are restricted to the provided baseline model and the given small support set, which may be used to fine-tune or guide the model for better transfer to the target domain. This track is designed to ensure a fair comparison of different adaptation algorithms.
open-source track: There are no restrictions on base models; even commercial models are encouraged to evaluate their performance on our challenging out-of-domain targets. Additional data (as long as it is not manually constructed to align specifically with the target domain) may be used for training, together with our provided support set.
Note that the current link is for source-limited track.

Important Dates
Challenges Leaderboards Open: 23 Feb 2026
Challenges Leaderboards Close: 13 May 2026
Challenges Technical Reports Deadline (on CMT) 20 May 2026
Notification to Challenge Winners: 27 May 2026
Challenge Reports ArXiv Deadline: 1 June 2026
Extended Abstract Deadline (on CMT): 27 April 2026
Extended Abstract Notification to Authors 18 May 2026
Extended Abstracts ArXiv Deadline 25 May 2026
Workshop Date: TBD
Note that if any changes happen, please follow the important dates reported in EgoVis

Statistics of Testing Domains
Domain Description Number of Test Questions
Surgery Laparoscopic surgery videos from surgeon's perspective 283
Industry Industrial assembly operations from worker's perspective 245
XSports Extreme sports (FPV drone racing, cycling, etc.) 246
Animal Pet-mounted camera footage from animal's perspective 183
Total - 957
Question Types
Identification (398 questions): Recognize objects, instruments, actions, or entities
Localization (284 questions): Locate objects in spatial or temporal dimensions
Prediction (161 questions): Predict future actions or events
Counting (114 questions): Count objects or occurrences
Tasks and Metrics
Multi-choice VQA is evaluated, and accuracy is used as the metric.

The leaderboard will show accuracies on all targets and also the overall accuracy, we will take the overall accuracy to rank models.

Provided Resources

1. Testing Set
   Test set could be downloaded here: https://huggingface.co/datasets/myuniverse/EgoCross

2. Training Support Set
   -In addition to the novel testing data, we further provide an extra support set, which is composed by few labeled examples for each domain. Particants could use these data to enhance model's performance.

Support set could be downloaded here: https://modelscope.cn/datasets/YuLi2024/EgoCross_support_set or here: https://huggingface.co/datasets/myuniverse/EgoCross

3. Baseline
   For this source-limited track, to strictly ensure the fairness of comparisions, we are set limitation on the baseline model.

Blancing both performance and compuation cost, QWen3-VL-4B is eventually selected as the baseline.

We provide a SFT verison of QWen3-VL-4B to help you quickly get one result. Baselines codes could be found: https://github.com/LiYu0524/EgoCross_SFT_qwen3vl4b. Feel free to use it or build your own codebase.

Rules & Guidelines
To ensure fairness and meaningful benchmarking, participants must adhere to the following rules:

✅ Allowed:

Participants may use our provided support set.

Participants may use publicly available datasets.

Novel algoritms are greatly encouraged.

❌ Not Allowed:

Replacing the base model to more advanced ones is not allow for this track.

Manually searching for and constructing additional support examples beyond the given setting is not allowed.

In the final phase, participants are required to clearly state its proposed method, including baseline model (should be QWen3-VL-4B in this track), used data, method framework, and training & inference details.
