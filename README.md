# ActiveVertexModel
An active vertex model (AVM) implementation on a periodic domain in Python, following the method outlined in Ref. [1]. For a full description of the AVM and its implementation please see [1] and its accompanying SI.

The main body of the code, where parameters are chosen, can be found in 'main' with auxillary functions found in 'functions'. The function that performs cell rearrangements ('UpdateTopology') uses further functions that can be found in 'T1functions'.

The model saves the coordinates of the cell vertices and cell centres of mass. It also saves the vertex coordinates in a form such that the outlines of the cells can be plotted using gnuplot. The script 'cell_plotter' then uses these files to plot the cell layer. These figures can then be animated using 'animator'.

[1] A. Killeen, T. Bertrand and C. F. Lee, Polar Fluctuations Lead to Extensile Nematic Behavior in Confluent Tissues, Physical Review Letters, 128 (7), 078001, 2022

contact: a.killeen18@imperial.ac.uk
