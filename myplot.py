import matplotlib.pyplot as plt


def mplt(x_list, y_list, legends, x_label = 'x', y_label = 'y', \
        title = 'title', linewidth = 3, figsize = (9, 6)):
    plt.figure(figsize = figsize)
    for i in range(len(x_list)):
        plt.plot(x_list[i], y_list[i], label = legends[i], linewidth = linewidth)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig(title + '.svg')
    pass
