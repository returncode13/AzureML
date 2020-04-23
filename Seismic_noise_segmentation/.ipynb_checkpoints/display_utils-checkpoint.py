import obspy
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import hvplot.xarray
import holoviews as hv
from holoviews.operation.datashader import datashade, rasterize
from holoviews import opts
from holoviews.operation.datashader import dynspread

hv.extension('bokeh')
hv.output(backend='bokeh')
#opts.Image(invert_yaxis=True,cmap='gray',height=1000,width=1000)
def shot_display(data=None,time=None,channels=None):
    #Create xrray dataset
    xarr=xr.Dataset(
        {
            'amplitudes':(('channels','time'),data.T)
        },
        {
            "channels":channels,"time":time
        }
    )
    
    #create an holoviews Dataset
    hvds=hv.Dataset(xarr)
    
    #create a holoviews Image element
    hvimage=hv.Image(hvds)
    return hvimage.opts(opts.Image(invert_yaxis=True,cmap='gray',height=1000,width=1000))
    #return hvimage
    
    