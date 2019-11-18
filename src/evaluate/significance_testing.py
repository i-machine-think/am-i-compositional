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

    results1 = args["results1"]
    results2 = args["results2"]
    one_sided = args["one_sided"]

    p = p_test(results1, results2, one_sided)
    print(f"Performance model 1: {results1}")
    print(f"Performance model 2: {results2}")
    print(f"T-test with one-sided = {one_sided}, p-value = {p}")
