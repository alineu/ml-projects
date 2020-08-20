import collections
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from wordcloud import WordCloud, STOPWORDS

def get_cmap(n, name='viridis'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''

    return plt.cm.get_cmap(name, n)


def plot_conf_mat(ax, confusion_mat, target_names, title_name,
                  colmap='Blues', alpha_=1, normalize=True, fontSize=18):
    '''Adopted from mlxtend'''

    accuracy = np.trace(confusion_mat) / float(np.sum(confusion_mat))
    misclassified = 1 - accuracy
    cmap = get_cmap(len(target_names)**2, colmap)
    ax.imshow(confusion_mat, interpolation='nearest', cmap=cmap, alpha=alpha_)
    ax.set_title('Confusion matrix of {0:s}'.format(
        title_name), fontsize=fontSize)
    # set ticks and ticklabels (optional)
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        ax.set_xticklabels(target_names, rotation=45,
                           fontsize=int(fontSize / 1.3))
        ax.set_xticks(tick_marks)
        ax.set_yticklabels(target_names, fontsize=int(fontSize / 1.3))
        ax.set_yticks(tick_marks)
    # normalize (optional)
    if normalize:
        confusion_mat = confusion_mat.astype('float')\
            / confusion_mat.sum(axis=1)[:, np.newaxis]
    thresh = confusion_mat.max() / 1.5 if normalize else confusion_mat.max() / 2
    # confusion matrix adjustments
    for i, j in itertools.product(range(confusion_mat.shape[0]),
                                  range(confusion_mat.shape[1])):
        if normalize:

            ax.text(j, i, "{:0.4f}".format(confusion_mat[i, j]),
                    horizontalalignment="center",
                    color="white" if confusion_mat[i, j] > thresh else "black",
                    fontsize=int(fontSize / 1.5))
        else:
            ax.text(j, i, "{:,}".format(confusion_mat[i, j]),
                    horizontalalignment="center",
                    color="white" if confusion_mat[i, j] > thresh else "black",
                    fontsize=int(fontSize / 1.5))

    ax.set_ylabel('True label', fontsize=int(fontSize))
    ax.set_xlabel('Predicted label\naccuracy={:0.2f}%; misclassified={:0.2f}%'.
                  format(100 * accuracy, 100 * misclassified), fontsize=int(fontSize))


def corr(corr, target_names=None, title_name='title', fig_size=(10, 12),
         colmap='Blues', alpha_=1, fontSize=18, masked=True):

    fig = plt.figure(figsize=fig_size)
    gs = fig.add_gridspec(10, 100)
    ax = fig.add_subplot(gs[:, :-5])
    cmap = get_cmap(corr.shape[0]**2, colmap)
    np.ones([3, 3], dtype=bool)
    mask = np.zeros_like(corr, dtype=np.bool)
    if masked:
        mask[np.triu_indices_from(mask)] = True
    else:
        mask[:] = False
    corr_im = np.ma.masked_where(mask, corr)
    ax.imshow(corr_im, interpolation='nearest', cmap=cmap, alpha=alpha_)
    ax.set_title('Correlation matrix of %s' % title_name, fontsize=fontSize)

    thresh = corr.max() / 1.5
    for i, j in itertools.product(range(corr.shape[0]),
                                  range(corr.shape[1])):
        ax.text(j, i, "{:0.2f}".format(corr_im[i, j]),
                horizontalalignment="center",
                color="white" if corr_im[i, j] > thresh else "black",
                alpha=0 if mask[i, j] else 1,
                fontsize=int(fontSize / 1.5))
    ax.set_xticks([])
    ax.set_yticks([])

    if target_names is not None:
        for i in range(len(target_names)):
            ax.text(-0.625, i, "{0:s}".format(target_names[i]),
                    horizontalalignment="right",
                    color="black", fontsize=int(fontSize / 1.5))
            ax.text(i, len(target_names) - 0.375, "{0:s}".format(target_names[i]),
                    verticalalignment="top", rotation=-45,
                    color="black", fontsize=int(fontSize / 1.5))

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cbar_ax = fig.add_subplot(gs[1:-1, -2:])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=colmap),
                      cax=cbar_ax, orientation='vertical',
                      label='', alpha=1)
    cb.ax.set_ylabel(cb.ax.get_xlabel(), fontsize=12)
    cb.ax.yaxis.set_tick_params(labelsize=fontSize)
    cb.outline.set_visible(False)
    plt.show()
    
def hist(
        x, ax=None, fig=None, colmap="winter", density=True, labels=[], colbar = True,
        title='title', show_title=True, xlabel='x-label', ylabel=None,
        fontSize=20, keep_y_lab=True, rot=0, sort_ed=False, show_ylab=False):

    #     if len(x)!=len(labels):
    #         freq = collections.Counter(x)
    #         sorted_dict = {k: v for k, v in sorted(freq.items(), key=lambda item: item[1])}
    #         sorted_keys = list(sorted_dict.keys())
    if show_ylab:
        if ylabel == None:
            ylabel = 'normalized frequency' if density else 'count'
    width = 0.25  # must be <1
    unique_x = np.arange(1, len(labels) + 1)
    color_index = np.argsort(x)
    my_cmap = cm.get_cmap(colmap)
    my_norm = Normalize(0, vmax=max(x))
    
    if not density:
        if colbar:
            for i in range(len(unique_x)):
                ax.bar(unique_x[i], x[i], width, color=my_cmap(my_norm(x[i])))
        else:
            ax.bar(unique_x, x, width)
    else:
        if colbar:   
            for i in range(len(unique_x)):
                ax.bar(unique_x[i], np.count_nonzero(
                    x == x[i]) / len(x), width, color=my_cmap(my_norm(x[i])))
        else:
            ax.bar(unique_x, np.count_nonzero(x == unique_x) / len(x), width)
            
    ax.set_xticks(unique_x)
    ax.set_xticklabels(labels, rotation=rot,
                       horizontalalignment={0: 'center', 45: 'right', 90: 'center'}.get(rot))
    ax.set_xlabel(xlabel, fontsize=int(fontSize * 1.2))
    yticks = ax.get_yticks()
    if not keep_y_lab:
        ax.set_yticks([])
        ax.set_ylabel(None)
    else:
        ax.set_ylabel(ylabel, fontsize=int(fontSize * 1.2))
        ax.set_yticklabels([np.round(x, 2) for x in yticks])
        ax.yaxis.set_tick_params(labelsize=fontSize)

    ax.xaxis.set_tick_params(labelsize=fontSize)
    if show_title:
        ax.set_title(title, fontsize=int(fontSize * 1.2))

    if colbar:
        ax.set_ylim([0, 1.1*max(x)])
        bbox = ax.get_position()
        x0_new = bbox.bounds[0] + bbox.bounds[2] / 4
        dx = bbox.bounds[2] / 2
        y0_new = bbox.bounds[1] + 0.9 * bbox.bounds[3]
        dy = 0.05 * bbox.bounds[3]
        cbar_ax = fig.add_axes([x0_new,y0_new,dx,dy])
        cb = fig.colorbar(cm.ScalarMappable(norm=my_norm, cmap=my_cmap),
                          cax=cbar_ax, orientation='horizontal',
                          label=ylabel,alpha=1)
        cb.ax.set_xlabel(cb.ax.get_xlabel(),fontsize=fontSize)
        cb.ax.xaxis.set_tick_params(labelsize=fontSize)
        cb.outline.set_visible(False)
        ax.set_yticks([])
        ax.set_ylabel(None)
        
def cloud_of_words(text_df, mask=None, fig_size=(25, 5), collocations=True,
                   max_words=50, width=2500, height=500, include_numbers=False,
                   stopWords=None, relative_scaling=0.4, title='word cloud', 
                   title_size=75, random_state=None, max_font_size=200, colormap=None):

    wordcloud = WordCloud(stopwords=stopWords, max_words=max_words, collocations=collocations,
                          max_font_size=max_font_size, random_state=random_state,
                          width=width, height=height, mask=mask, include_numbers=include_numbers,
                          colormap=colormap, relative_scaling=relative_scaling).generate(str(text_df))

    plt.figure(figsize=fig_size)
    plt.imshow(wordcloud)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()