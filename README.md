# lightcurves

This is code I am writing at Oregon State University in 2016/2017 for analysis of the off-axis lightcurve from a GRB.
It's composed of two pieces:  `altonbrown.py` and `lightcurves.py`.  `altonbrown.py` returns the 3-dimensional coordinates of
the equal arrival time surface (EATS) of the light from the GRB.  This surface is then used by `lightcurves.py` in calculating
the expected lightcurve.

The calculation of the EATS follows [Panaitescu & Meszaros, 1998](http://iopscience.iop.org/article/10.1086/311127/pdf),
and the lightcurve calculation is from [Rossi et al., 2002](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/mnras/354/1/10.1111/j.1365-2966.2004.08165.x/2/354-1-86.pdf?Expires=1488744670&Signature=atMX7n6N-QpDgrp6yd7Q0VNgVMfN-GMi33gVNS43oRcr54UpNGg1nBeAHEoJoYo14jC7dosv-goFyh7tCeYfwB~anF2qsDnM-6Cqk749hgjPiUiFu77omIgCZHocRNcKCUgkzSpdaN1U9kG-0v1K5yv6EWfmsM1Rr-Lm0mePwpQIfmNOPfDEN6qBzTE-lL1VPG3v91VQt95-1B75J37BvSM64m2NdvR9z1cL~oNFM7M8rDJjg8FK-KfNdb35G7uDbivdnMAoKKmBlDOIXAZzY8qxfXF4VxeYbVJZhzlQ8nX0QX92-LfXUHN-oDloTMs~8RcNcwwsNg8Tf1barVVMuA__&Key-Pair-Id=APKAIUCZBIA4LVPAVW3Q).

Development is ongoing.
