import fire

from human_eval.data import HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness

def entry_point(sample_file: str, k=None, n_workers: int = 4, timeout: float = 3.0, problem_file: str = HUMAN_EVAL):
    if k is None:
        k = [1, 10, 100]
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)
    print(results)

def main():
    fire.Fire(entry_point)

if __name__ == "__main__":
    main()