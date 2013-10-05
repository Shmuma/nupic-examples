import math
import numpy
import pprint

from optparse import OptionParser

from matplotlib.pylab import draw, plot, ion, subplot, savefig

from nupic.frameworks.opf.modelfactory import ModelFactory

import model_params

class Plot(object):
    def __init__(self, error_window=360, window=-1):
        self.error_window = error_window
        self.window = window
        self.errors = []

        subplot(2, 1, 1)
        self.act_plot, = plot([], [])
        self.pred_plot, = plot([], [])
        subplot(2, 1, 2)
        self.err_plot, = plot([], [])

    def _apply_window(self, data):
        if self.window > 0:
            return data[-self.window:]
        else:
            return data

    def new_point(self, x, actual, predicted):
        xdata = self._apply_window(numpy.append(self.act_plot.get_xdata(orig=True), x))
        self.act_plot.set_xdata(xdata)
        self.act_plot.axes.set_xlim(min(xdata), x)

        dat = self._apply_window(numpy.append(self.act_plot.get_ydata(orig=True), actual))
        self.act_plot.set_ydata(dat)
        self.act_plot.axes.set_ylim(min(dat), max(dat))

        dat = self._apply_window(numpy.append(self.pred_plot.get_ydata(orig=True), predicted))
        self.pred_plot.set_xdata(xdata)
        self.pred_plot.set_ydata(dat)

        self.errors.append((actual - predicted)**2)
        self.errors = self.errors[-self.error_window:]
        if len(self.errors) > 5:
            rmse = math.sqrt(sum(self.errors)/len(self.errors))
        else:
            rmse = None

        self.err_plot.set_xdata(xdata)
        self.err_plot.axes.set_xlim(min(xdata), x)
        err_data = self._apply_window(numpy.append(self.err_plot.get_ydata(orig=True), rmse))
        self.err_plot.set_ydata(err_data)
        self.err_plot.axes.set_ylim(min(err_data), max(err_data))
        draw()


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-s", "--steps", dest="steps", type='int', default=1000,
                      help="Count of steps to simulate, default=1000")
    parser.add_option("-i", "--image", dest="image_name",
                      help="Name of image file to save final figure")
    parser.add_option("-w", "--window", dest="window", default=-1, type='int',
                      help="Data window size to show on charts. If greater than zero,"
                           "this count of last values displayed. By default display all values")
    options, args = parser.parse_args()

    ion()
    model = ModelFactory.create(model_params.MODEL_PARAMS)
    model.enableInference({'predictedField': 'y'})

    figure = Plot(window=options.window)

    x = 0
    prev_pred = None

    while x < options.steps:
        y = math.sin(x*math.pi/180.0)

        res = model.run({'y': y})
        inference = res.inferences['multiStepBestPredictions'][1]
        if prev_pred:
            figure.new_point(x, y, prev_pred)
        prev_pred = inference

        x += 1
    if options.image_name:
        savefig(options.image_name)
