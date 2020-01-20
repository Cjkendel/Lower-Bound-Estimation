from LocalVariationalBoundsVectorized import EstimateLowerBound
from LogisticRegression import LogisticReg

if __name__ == "__main__":
    lower = EstimateLowerBound(batch_size=False, full_batch=True, n_batch=False)
    params = lower.call_and_write_results(100)
    #
    # model = LogisticReg(batch_size=16, full_batch=False, n_batch=False)
    # model.call_and_write_results(5000)
