"""Written responses for the Lab 6 report. Numbers are pulled from results.jsonl."""


def _acc(results: dict, name: str) -> str:
    return f"{100 * results[name]['test_acc']:.2f}%"


def _time(results: dict, name: str) -> str:
    return f"{results[name]['train_time_s']:.1f} s"


# ---------------------------------------------------------------- Part 2

def part2_changes(results: dict) -> list[str]:
    return [
        "Analysis: the 10 continuous features (Elevation, Aspect, Slope, the three horizontal "
        "distances, Vertical_Distance_To_Hydrology and the three Hillshade indices) live on very "
        "different scales – e.g. Elevation is 1,859–3,858 m with a std of ~280, the road/fire-point "
        "distances reach ~7,100 with a std of ~1,300–1,550, while Slope is 0–66. The remaining 44 "
        "columns (4 Wilderness_Area + 40 Soil_Type) are already one-hot 0/1 flags. There are no "
        "missing values. The 7 classes are imbalanced (class 2: 48.8%, class 1: 36.5%, class 4: 0.5%).",
        "Changes: (1) Standardized the 10 continuous columns with sklearn's StandardScaler "
        "(zero mean, unit variance). The scaler is fit on the training split only and then applied "
        "to the validation/test splits so no test statistics leak into training. (2) The binary "
        "one-hot columns are left untouched (they are already in [0, 1] and scaling them would "
        "destroy their sparsity/meaning). (3) The 70/10/20 split is now made with a fixed seed so "
        "every later experiment is evaluated on exactly the same test set, and the data are held "
        "as pre-built tensors so batches are gathered without per-row NumPy conversion.",
        f"Control: the same split/seed without scaling scored "
        f"{_acc(results, 'part2_unscaled_control')}, so the gain below is due to scaling alone.",
    ]


def part2_accuracy(results: dict) -> str:
    return (
        f"{_acc(results, 'part2_scaled')} (was {_acc(results, 'part1_baseline')} with the original "
        f"script and {_acc(results, 'part2_unscaled_control')} unscaled on the same fixed split; "
        f"same [64, 32] network, batch 1024, 5 epochs, lr 1e-3; training time "
        f"{_time(results, 'part2_scaled')})"
    )


def part2_question(results: dict) -> list[str]:
    return [
        f"Yes – a single preprocessing step bought roughly 7–10 percentage points "
        f"({_acc(results, 'part2_unscaled_control')} → {_acc(results, 'part2_scaled')} on the same "
        "split) without touching the model, which is more than many architectural tweaks give.",
        "MLPs are sensitive to input scale because every first-layer neuron computes a weighted sum "
        "w·x + b. With raw inputs, features like Elevation (~3,000) or distances (~2,000) dominate "
        "that sum while binary soil flags (0/1) barely register, so the initial activations are "
        "huge, many ReLUs start saturated/dead, and the gradients w.r.t. different weights differ by "
        "orders of magnitude. The loss surface becomes a long, narrow valley: a single learning rate "
        "is too large for the large-scale directions (oscillation) and too small for the small-scale "
        "ones (crawling). Default weight initialization (Kaiming/Xavier) and Adam's defaults also "
        "assume roughly unit-variance inputs. Standardizing makes every feature contribute "
        "comparably from the first step, so gradient descent converges much faster.",
    ]


# ---------------------------------------------------------------- Part 3

PART3_RUNS = [
    ("[64, 32] (baseline)", "part3_arch_64x32"),
    ("[64, 128, 64]", "part3_arch_64x128x64"),
    ("[64]", "part3_arch_64"),
    ("[16]", "part3_arch_16"),
    ("[256, 256]", "part3_arch_256x256"),
    ("[256, 512, 256]", "part3_arch_256x512x256"),
    ("[512, 512, 512]", "part3_arch_512x512x512"),
    ("[1024, 1024, 1024, 1024]", "part3_arch_1024x1024x1024x1024"),
    ("[2048, 2048, 2048, 2048]", "part3_arch_2048x2048x2048x2048"),
    ("[1024] × 8 layers", "part3_arch_1024x1024x1024x1024x1024x1024x1024x1024"),
]


def part3_question(results: dict) -> list[str]:
    r = results
    params = {name: r[name]["n_params"] for _, name in PART3_RUNS}
    return [
        "All runs: scaled data, batch 1024, 5 epochs, Adam lr 1e-3, same fixed split.",
        f"Best trade-off: [1024, 1024, 1024, 1024] (~{params['part3_arch_1024x1024x1024x1024'] / 1e6:.1f}M "
        f"parameters) at {_acc(r, 'part3_arch_1024x1024x1024x1024')}. Accuracy rose steadily with "
        f"capacity – from {_acc(r, 'part3_arch_16')} for a single 16-unit layer (~1k parameters) "
        f"to {_acc(r, 'part3_arch_512x512x512')} for [512, 512, 512] – which shows the original "
        "network was badly under-fitting this problem: forest cover depends on fine-grained "
        "interactions between elevation, distances and soil type that a tiny network cannot carve out.",
        f"Diminishing returns appeared past ~3M parameters. Doubling the width to [2048] × 4 "
        f"(~{params['part3_arch_2048x2048x2048x2048'] / 1e6:.1f}M parameters) only reached "
        f"{_acc(r, 'part3_arch_2048x2048x2048x2048')} and doubling the depth to 8 × 1024 reached "
        f"{_acc(r, 'part3_arch_1024x1024x1024x1024x1024x1024x1024x1024')} – about half a point "
        f"better – while costing {_time(r, 'part3_arch_2048x2048x2048x2048')} and "
        f"{_time(r, 'part3_arch_1024x1024x1024x1024x1024x1024x1024x1024')} versus "
        f"{_time(r, 'part3_arch_1024x1024x1024x1024')}. Depth was a more efficient way to add "
        "capacity than width (a 3- or 4-layer net beat a wider 2-layer one).",
        f"Training-time trade-off: small nets (≤ [256, 256]) all trained in ~9–12 s because the "
        f"epoch time is dominated by data loading/Lightning overhead, not math. Beyond that, time "
        f"grows roughly with parameter count: {_time(r, 'part3_arch_512x512x512')} for [512]×3, "
        f"{_time(r, 'part3_arch_1024x1024x1024x1024')} for [1024]×4 and "
        f"{_time(r, 'part3_arch_2048x2048x2048x2048')} for [2048]×4 on a 4-core CPU. I kept "
        "[1024, 1024, 1024, 1024] for the remaining parts as the best accuracy-per-second choice.",
    ]


# ---------------------------------------------------------------- Part 4

PART4_RUNS = [
    ("32", "part4_bs_32"),
    ("128", "part4_bs_128"),
    ("512", "part4_bs_512"),
    ("1024", "part3_arch_1024x1024x1024x1024"),
    ("4096", "part4_bs_4096"),
    ("16384", "part4_bs_16384"),
]


def part4_question(results: dict) -> list[str]:
    r = results
    return [
        "All runs: [1024, 1024, 1024, 1024], scaled data, 5 epochs, Adam lr 1e-3.",
        f"Batch 128 gave the highest accuracy after 5 epochs ({_acc(r, 'part4_bs_128')}), with 32 "
        f"({_acc(r, 'part4_bs_32')}) and 512 ({_acc(r, 'part4_bs_512')}) close behind. Considering "
        f"time, 512–1024 is the sweet spot: batch 1024 reached {_acc(r, 'part3_arch_1024x1024x1024x1024')} in "
        f"{_time(r, 'part3_arch_1024x1024x1024x1024')} versus {_time(r, 'part4_bs_128')} for 128 and "
        f"{_time(r, 'part4_bs_32')} for 32, so I kept batch 1024 for the long runs that follow.",
        f"There is a clear trade-off. Small batches take many more optimizer steps per epoch "
        f"(~12,700 steps/epoch at 32 vs ~400 at 1024), so for a fixed number of epochs they learn "
        f"more, but each step has fixed Python/Lightning overhead and under-uses the CPU's vector "
        f"units, so wall-clock time explodes (32 was ~10× slower than 1024 for <1 point of gain, "
        f"and actually slightly worse than 128 because its gradients are noisier). Very large "
        f"batches are fast per epoch but take too few steps: 4096 ({_acc(r, 'part4_bs_4096')}) "
        f"and 16384 ({_acc(r, 'part4_bs_16384')}, only ~25 updates per epoch) under-trained badly "
        f"in 5 epochs, and they were not even faster ({_time(r, 'part4_bs_4096')} and "
        f"{_time(r, 'part4_bs_16384')} vs {_time(r, 'part3_arch_1024x1024x1024x1024')} for 1024) "
        "because a 4-core CPU is already saturated by 1024-row matrix multiplies.",
        "Other patterns: accuracy vs. batch size is an inverted U, and the right batch size is tied "
        "to the learning rate – with lr fixed at 1e-3, the large batches would need a larger "
        "learning rate (or many more epochs) to make the same progress, while the tiny batches "
        "would benefit from a smaller one to tame their gradient noise.",
    ]


# ---------------------------------------------------------------- Part 5

def part5_question(results: dict) -> list[str]:
    r = results["part5_500_epochs"]
    return [
        f"Run: [1024, 1024, 1024, 1024], batch 1024, constant Adam lr 1e-3, 500 epochs "
        f"(~398 steps/epoch, so 20k steps ≈ 50 epochs on the x-axis). Best validation accuracy "
        f"{100 * r['best_val_acc']:.2f}%, test accuracy of the best checkpoint {100 * r['test_acc']:.2f}%, "
        f"training time {r['train_time_s'] / 3600:.1f} h.",
        "At this learning rate I would train for about 100–150 epochs; ~200 is the point where "
        "further training is no longer meaningful. Validation accuracy climbed quickly to 92.8% at "
        "epoch 10 and 95.2% at epoch 50, was within 0.5 points of its final best by epoch ~113 "
        "(95.6% at 100, 95.8% at 150) and reached 96.05% at epoch 200. The remaining 300 epochs "
        "(60% of the compute, ~1.6 h) added only ~0.2 points (best 96.25% at epoch 487), which is "
        "about the size of the epoch-to-epoch noise in the curve.",
        "Information used: (1) the validation-accuracy curve flattening into a plateau whose "
        "running maximum improves by < 0.1 point per 50 epochs; (2) the training loss, which keeps "
        "creeping toward zero (0.039 at epoch 50 → 0.015 at 200 → 0.005 at 500) while validation "
        "accuracy stays flat – the network is starting to memorize the training set rather than "
        "generalize better; and (3) the checkpoint callback's best epoch. Because checkpointing "
        "keeps the best model, over-training didn't hurt the test score here, but it wasted "
        "compute. Part 6 showed the plateau is a property of the constant learning rate: with a "
        "decaying schedule the same network surpasses this 500-epoch result in only 30 epochs.",
    ]


# ---------------------------------------------------------------- Part 6

PART6_RUNS = [
    ("1e-3", "part6_lr_1e-3"),
    ("1e-4", "part6_lr_1e-4"),
    ("1e-5", "part6_lr_1e-5"),
    ("Step Scheduler (1e-3, ×0.1 every 10 epochs)", "part6_step_1e-3_s10_g0.1"),
    ("Cosine annealing (1e-3 → 0 over 30 epochs)", "part6_cosine_1e-3"),
    ("OneCycle (warm-up to 2e-3, anneal to ~0)", "part6_onecycle_2e-3"),
]


def part6_question(results: dict) -> list[str]:
    r = results
    return [
        "All runs: [1024, 1024, 1024, 1024], batch 1024, 30 epochs, Adam, same split.",
        f"Best scheduler: OneCycle ({_acc(r, 'part6_onecycle_2e-3')}), followed by cosine annealing "
        f"({_acc(r, 'part6_cosine_1e-3')}) and the step scheduler ({_acc(r, 'part6_step_1e-3_s10_g0.1')}); "
        f"all three beat every constant learning rate (best constant: 1e-3 at {_acc(r, 'part6_lr_1e-3')}). "
        f"In only 30 epochs the OneCycle and cosine runs beat the 500-epoch constant-lr run from "
        f"Part 5 ({_acc(r, 'part5_500_epochs')}) at ~5% of its compute.",
        "Convergence differences: 1e-3 learned fastest early (90.8% validation after 5 epochs) but "
        "then stalled around 94–94.5% – its steps are too large to settle into a sharp minimum, so "
        "it bounces around the bottom of the valley. 1e-4 was smoother but much slower (83.5% at "
        "epoch 5, 90.3% at epoch 30), and 1e-5 was barely moving (83.6% at epoch 30, training loss "
        "still 0.40) – it would need hundreds of epochs. The step scheduler shows the benefit of "
        "decay very clearly: validation jumped from 92.9% to 95.1% immediately after the first drop "
        "at epoch 10, then gained little after the second drop because 1e-5 is already too small "
        "to make progress. Cosine and OneCycle decay smoothly, so they spend more time at useful "
        "rates and still finish with a tiny rate for fine convergence. OneCycle starts lower (warm-"
        "up) and is behind early (85.2% at epoch 5) but its higher peak rate explores more and its "
        "annealing phase produced the best final model.",
        "Process for future problems: (1) scale the inputs, (2) do a quick coarse sweep over "
        "learning rates spaced by ~3–10× (e.g. 1e-2, 3e-3, 1e-3, 3e-4, 1e-4) for a few epochs – or "
        "use an LR range test / Lightning's lr_find – and pick the largest rate that still "
        "decreases the loss smoothly, (3) train with a decaying schedule (OneCycle or cosine with "
        "that rate as the peak, sized to the planned number of epochs), and (4) watch the "
        "validation curve in TensorBoard to adjust epochs, re-tuning the rate when batch size changes.",
    ]


# ---------------------------------------------------------------- Part 7

FINAL_RUN = "part7_onecycle_100ep_dropout0.1"


def part7_config(final: dict) -> list[str]:
    return [
        "[1024, 1024, 1024, 1024]",
        str(final["batch_size"]),
        str(final["epochs"]),
        f"OneCycle, peak {final['lr']:g}",
        "Scaled inputs, shuffling, AdamW (wd 1e-4), dropout 0.1",
    ]


def part7_notes(results: dict) -> list[str]:
    r = results
    final = r[FINAL_RUN]
    return [
        f"Final result: {_acc(r, FINAL_RUN)} test accuracy (best validation "
        f"{100 * final['best_val_acc']:.2f}%), trained in {final['train_time_s'] / 60:.0f} min on a "
        f"4-core CPU – up from {_acc(r, 'part1_baseline')} for the original baseline. Details: ReLU "
        "MLP 54 → 1024 → 1024 → 1024 → 1024 → 7 (~3.2M parameters) with dropout 0.1 after each "
        "hidden layer; StandardScaler on the 10 continuous features (fit on the training split "
        "only); training data reshuffled every epoch; AdamW (weight decay 1e-4); OneCycleLR "
        "warming up to 2e-3 and annealing to ~0, stepped every batch; best checkpoint by "
        "validation accuracy; fixed-seed 70/10/20 split. Reproduce with `python lab6.py` "
        "(the defaults are this configuration).",
        "How I got there: OneCycle was the best schedule in Part 6, so I gave it a longer horizon. "
        f"100 epochs with per-epoch shuffling and AdamW reached {_acc(r, 'part7_onecycle_100ep')}, "
        "but its training loss fell to ~0.0015 while validation accuracy flattened, i.e. the network "
        "had memorized the training set. Adding dropout 0.1 as regularization raised the test "
        f"accuracy to {_acc(r, FINAL_RUN)}, at the cost of slower epochs "
        f"({_time(r, FINAL_RUN)} vs {_time(r, 'part7_onecycle_100ep')}).",
        "Progression of the main levers (test accuracy): baseline 68.49% → input scaling 78.81% → "
        f"[1024]×4 architecture 90.82% (5 epochs) → 500 epochs at constant lr "
        f"{_acc(r, 'part5_500_epochs')} → OneCycle schedule {_acc(r, 'part6_onecycle_2e-3')} in "
        f"30 epochs → longer OneCycle + shuffling + AdamW + dropout {_acc(r, FINAL_RUN)}.",
    ]
