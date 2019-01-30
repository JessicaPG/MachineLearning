from With_Testing import RegressionTaskWithTesting, ClassificationTaskWithTesting
import sys


def main():
    try:
        RegressionTaskWithTesting
        ClassificationTaskWithTesting
        return 0
    except:
        return 1

if __name__ == "__main__":
    sys.exit(main())