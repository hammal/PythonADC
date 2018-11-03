"""
Auxiliary functions for saving all open simulations and extracting arguments
"""

import argparse
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import sys
import os


parser = argparse.ArgumentParser(description='Analog-to-digital processing tool')
parser.add_argument('-f', '--filename', type=str,
                   help='Supply filename to save all figures in', default=sys.argv[0])
# parser.add_argument('--sum', dest='accumulate', action='store_const',
                #    const=sum, default=max,
                #    help='sum the integers (default: find the max)')

args = parser.parse_args()

def SaveFigures(figs=None, dpi=200):
    if args.filename:
        pp = PdfPages(args.filename + ".pdf")
    else:
        pp = PdfPages(os.path.splitext(sys.argv[0])[0] + ".pdf")
    
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()