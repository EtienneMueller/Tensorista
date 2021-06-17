class SparseCategoricalCrossentropy:
    def __init__(self):
        print("SparseCategoricalCrossentropy chosen")

# Binary Cross Entropy Loss
# Loss = (1/batch_size) * np.sum(-(Y * np.log(self.A[1]) + (1-Y) * np.log(1-self.A[1])))
