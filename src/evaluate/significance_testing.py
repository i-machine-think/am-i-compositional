import argparse
from scipy import stats


def p_test(results1, results2, one_sided=True):
    t, p = stats.ttest_ind(results1, results2, equal_var=False)
    if one_sided:
        p /= 2
    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results1", type=float, nargs="+",
                        help="Lists of floats, performance of model 1.")
    parser.add_argument("--results2", type=float, nargs="+",
                        help="Lists of floats, performance of model 2.")
    parser.add_argument("--one_sided", action="store_true")
    args = vars(parser.parse_args())

    p_test(args["results1"], args["results1"], args["one_sided"])
