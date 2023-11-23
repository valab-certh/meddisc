from matplotlib import pyplot as plt
import numpy as np


class DetectionVisuals:

    def __init__(self, fig_title, n_axes = 3):

        assert n_axes in {2, 3}, 'E: n_axes must be 2 or 3.'
        self.n_axes = n_axes
        if self.n_axes == 2:
            figsize = (12, 6.3)
            self.ax_titles = ('Input DICOM image', 'Cleaned image')
        elif self.n_axes == 3:
            figsize = (12, 5)
            self.ax_titles = ('Input DICOM image', 'Rescaled Bboxes', 'Cleaned image')
        self.fig, self.axes = plt.subplots(nrows = 1, ncols = self.n_axes , figsize = figsize)
        self.fig_title = fig_title

    def build_plt(self, imgs: list[np.ndarray], removal_period: float):
        '''
            Args:
                imgs. Length 2 or 3.
                    imgs[i]. Shape (height, width).
                    imgs[0]. Input image.
                    imgs[1]. Contour image. (Optional)
                    imgs[2]. Cleaned image.
        '''

        assert len(imgs) == self.n_axes, 'E: len(imgs) must be %d.'%self.n_axes

        for ax_idx, (ax, img) in enumerate(zip(self.axes, imgs)):
            ax.imshow(X = img, cmap = 'gray')
            ax.set_title(self.ax_titles[ax_idx])
            ax.axis('off')

        self.fig.suptitle(self.fig_title + '\nImage Resolution: (%d, %d)'%(img.shape[0], img.shape[1]) + '\nRemoval Period: %.4f'%(removal_period))

    def display():
        plt.show()