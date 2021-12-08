EO times series.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A brief demo of extracting time series for a point shapefile and writing as attributes. This is not exhaustive, simply what has been required of a recent project. Two of these require CEDA credentials to access data and the S1/S2 requires a google earth engine account. 

**Met_data_tseries.ipynb**

Attribute a geometry file with met office climate modelling data from the CEDA archive. 

**S1_&_S2time_series.ipynb**

Attribute a geometry file with S2 NDVI and S1 GRD-based time series. 

**Nextmapprocessing.ipynb**

Attribute a geometry file with Nextmap elevation data and derivatives from the CEDA archive. 

I have just provided cut down modules locally in the directory src. Provided you are using the notebook herein you can import the functions from the local files.

Installation and Use
~~~~~~~~~~~~~~~~~~~~

Installing the required libs (there are not many) uses the conda system so ensure you have this first. Clone this repo and cd into the directory then...

.. code-block:: bash

conda env create -f eot_demo.yml

conda activate eot

jupyter notebook


Then open the ipynb and cycle through the cells.

Notes
~~~~~

You may note the functions spread the tasks client-side rather than server-side. Having written both, it becomes a case of moving the bottleneck between either client threads or file download from google if using server side parallelism. At present the file download is unreliable from google, and takes about the same length of time as spreading the process client side, hence the reliable clientside approach is used. Improvements on this welcome!
