from utils.parser import build_parser
from utils.train_helpers import train
from utils.eval_helpers import evaluate_all_epochs

# TODO: function space filtration on test + test corrupt
# TODO: train set evaluation for epistemic uncertainty + test set for aleatoric uncertainty
# TODO: weight space filtration over time (delta?)
# TODO: cov mat as distance matrix or PCD?
# TODO: eNTK?


def main():
    """
    Main function for training MultiSWAG model and evaluating on ID and OOD validation datasets.
    """
    # TODO: evaluate on train set for epistemic uncertainty
    # TODO: evaluate on test set for aleatoric uncertainty
    # Note: p_params is list of [tensor(num_models, model_params, layer)]
    args = build_parser().parse_args()
    # Print args
    print(args)

    if args.mode == "train":
        train(args)
    if args.mode == "eval":
        evaluate_all_epochs(args)
    if args.mode == "analysis":
        pass

    # Pac Bayes vs Bayesian, is this paper testing agaisnt multiswag or just MCMC?
    # https://arxiv.org/html/2406.05469v1#S3


if __name__ == "__main__":
    main()
