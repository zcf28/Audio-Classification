import os
import matplotlib
import matplotlib.pyplot as plt


def draw_progress(train_loss, val_loss, val_error_rate,
                  curr_epoch,
                  cv, dir_to_save):

    matplotlib.use('Agg')

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(train_loss, "r", label='Train loss')
    plt.plot(val_loss, "g", label='Val loss')
    plt.title(cv)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(val_error_rate, "g", label='Test error rate')
    plt.title(cv)
    plt.xlabel("epoch")
    plt.ylabel("error rate")
    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(dir_to_save, cv)+".pdf")
