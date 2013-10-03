import math
import time
import random

from matplotlib.pylab import draw, plot, ion, subplot, savefig

from nupic.frameworks.opf.modelfactory import ModelFactory

import model_params

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
    # error
    edata = []

    subplot(2, 1, 1)
    sin_plot, = plot(xdata, ydata)
    sin_plot.axes.set_ylim(-1.1, 1.1)
    pred_plot, = plot(xdata, pdata)
    pred_plot.axes.set_ylim(-1.1, 1.1)

    subplot(2, 1, 2)
    err_plot, = plot(xdata, edata)

    x = 0
    prev_pred = None

    while x < 2000:
        y = math.sin(x*math.pi/180.0)# + math.sin(20.0*x*math.pi/180.0)/10.0
        if x > 1000:
            if y > 0:
                y = min(0.5, y)
#            else:
#                y = max(-0.5, y)
        res = model.run({'y': y})
        inference = res.inferences['multiStepBestPredictions'][1]

        pdata.append(prev_pred)
        xdata.append(x)
        ydata.append(y)
        if prev_pred:
            edata.append(abs(y-prev_pred))
        else:
            edata.append(None)

        prev_pred = inference

        if x % 100 == 0:
            sin_plot.set_xdata(xdata)
            sin_plot.set_ydata(ydata)
            sin_plot.axes.set_xlim(0, x)
            pred_plot.set_xdata(xdata)
            pred_plot.set_ydata(pdata)

            err_plot.set_xdata(xdata)
            err_plot.set_ydata(edata)
            err_plot.axes.set_xlim(0, x)
            err_plot.axes.set_ylim(min(edata), max(edata))
            draw()
        x += 1

    savefig("figure.png")