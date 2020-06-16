import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')


def save_plots(train_losses, train_accs, test_losses, test_accs, file_prefix=''):
    """Plot

        Plot two figures: loss vs. epoch and accuracy vs. epoch
    """
    n = len(train_losses)
    xs = np.arange(n)

    # plot losses
    fig, ax = plt.subplots()
    ax.plot(xs, train_losses, '--', linewidth=2, label='train')
    ax.plot(xs, test_losses, '-', linewidth=2, label='validation')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.legend(loc='upper right')
    plt.savefig('./figures/{}loss.png'.format(file_prefix + '_'))

    # plot train and test accuracies
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(xs, train_accs, '--', linewidth=2, label='train')
    ax.plot(xs, test_accs, '-', linewidth=2, label='validation')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.legend(loc='lower right')
    plt.savefig('./figures/{}accuracy.png'.format(file_prefix + '_'))

    plt.clf()
    plt.close('all')
