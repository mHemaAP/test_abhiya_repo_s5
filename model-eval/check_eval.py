import os
import sys

def main():
    file_path = './model/eval_results.json'  # Or wherever you save the evaluation results

    if os.path.isfile(file_path):
        print("Evaluation finished successfully.")
        sys.exit(0)  # Exit with status 0, meaning everything is ok
    else:
        print("Evaluation not finished.")
        sys.exit(1)  # Exit with status 1, meaning there was a problem

if __name__ == "__main__":
    main()
