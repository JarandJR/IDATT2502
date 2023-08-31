from non_linear_regression_2d import Visulization

if __name__ == "__main__":
    visulizer = Visulization(file='oving1/taskC/day_head_circumference.csv')
    visulizer.train_and_visualize(epochs=200_000, learning_rate=0.000001)
    