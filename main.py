from LocalVariationalBoundsVectorized import EstimateLowerBound
from LogisticRegression import LogisticReg

if __name__ == "__main__":
    #JJ Bound Estimation
    lower = EstimateLowerBound(batch_size=1, full_batch=False, n_batch=False)
    # lower.call_and_write_results(10000)


    # Torch Logistic Regression, get point estimates
    # model = LogisticReg(batch_size=16, full_batch=False, n_batch=False)
    # model.call_and_write_results(5000)

    # Graph means over time
    for i in range(1, 5):
        lower.generate_plots([1, 2, 4, 8, 16, 32, lower.trainloader.total_data_len], epochs=i)

