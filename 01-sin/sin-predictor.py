import math
import numpy
import time
import random

from matplotlib.pylab import draw, plot, ion, subplot, savefig

from nupic.frameworks.opf.modelfactory import ModelFactory

import model_params

class Plot(object):
    def __init__(self, error_window=360):
        self.error_window = error_window
        self.errors = []

        subplot(2, 1, 1)
        self.act_plot, = plot([], [])
        self.pred_plot, = plot([], [])
        subplot(2, 1, 2)
        self.err_plot, = plot([], [])

    def new_point(self, x, actual, predicted):
        xdata = numpy.append(self.act_plot.get_xdata(orig=True), x)
        self.act_plot.set_xdata(xdata)
        self.act_plot.axes.set_xlim(0, x)

        dat = numpy.append(self.act_plot.get_ydata(orig=True), actual)
        self.act_plot.set_ydata(dat)
        self.act_plot.axes.set_ylim(min(dat), max(dat))

        dat = numpy.append(self.pred_plot.get_ydata(orig=True), predicted)
        self.pred_plot.set_xdata(xdata)
        self.pred_plot.set_ydata(dat)

        self.errors.append((actual - predicted)**2)
        self.errors = self.errors[-self.error_window:]
        if len(self.errors) > 5:
            rmse = math.sqrt(sum(self.errors)/len(self.errors))
        else:
            rmse = None

        self.err_plot.set_xdata(xdata)
        self.err_plot.axes.set_xlim(0, x)
        err_data = numpy.append(self.err_plot.get_ydata(orig=True), rmse)
        self.err_plot.set_ydata(err_data)
        self.err_plot.axes.set_ylim(min(err_data), max(err_data))
        draw()


if __name__ == "__main__":
    ion()
    model = ModelFactory.create(model_params.MODEL_PARAMS)
    model.enableInference({'predictedField': 'y'})

    # x values
    xdata = []
    # sin values
    ydata = []
    # predicted values
    pdata = []
    # rmse
    edata = []

    error_window = 360
    errors = []

    figure = Plot()

#    subplot(2, 1, 1)
#    sin_plot, = plot(xdata, ydata)
#    sin_plot.axes.set_ylim(-1.1, 1.1)
#    pred_plot, = plot(xdata, pdata)
#    pred_plot.axes.set_ylim(-1.1, 1.1)

#    subplot(2, 1, 2)
#    err_plot, = plot(xdata, edata)

    x = 0
    prev_pred = None

    while x < 1000:
        y = math.sin(x*math.pi/180.0)
#        if x > 1000:
#            if y > 0:
#                y = min(0.5, y)
#            else:
#                y = max(-0.5, y)
        res = model.run({'y': y})
        inference = res.inferences['multiStepBestPredictions'][1]
        if prev_pred:
            figure.new_point(x, y, prev_pred)

        pdata.append(prev_pred)
        xdata.append(x)
        ydata.append(y)
        if prev_pred:
            edata.append(abs(y-prev_pred))
        else:
            edata.append(None)

        prev_pred = inference

 #       if x % 100 == 0:
 #           sin_plot.set_xdata(xdata)
 #           sin_plot.set_ydata(ydata)
 #           sin_plot.axes.set_xlim(0, x)
 #           pred_plot.set_xdata(xdata)
 #           pred_plot.set_ydata(pdata)

#            err_plot.set_xdata(xdata)
#            err_plot.set_ydata(edata)
#            err_plot.axes.set_xlim(0, x)
#            err_plot.axes.set_ylim(min(edata), max(edata))
#            draw()
        x += 1

    savefig("figure.png")