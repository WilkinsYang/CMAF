import matplotlib.pyplot as plt


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [TS]')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_distribution(test_labels, test_predictions):
    a = plt.axes(aspect='equal')
    #plt.grid()
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [TS]', fontsize=14)
    plt.ylabel('Predictions [TS]', fontsize=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    lims = [-60, -20]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()

def plot_histogram(error):
    plt.hist(error, bins=20,range=(-8.0,8.0))
    plt.xlabel('Prediction Error [TS]', fontsize=14)
    _ = plt.ylabel('Count', fontsize=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.show()