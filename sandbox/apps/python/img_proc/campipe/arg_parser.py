from __init__ import *

import optparse
import sys

sys.path.insert(0, ROOT+'apps/python/')

from pipe_options import *

def parse_args():

    help_str = \
    '"new" : from scratch | "existing" : compile and run |  "ready" : just run'
    parser.add_option('-m', '--mode',
                      type='choice',
                      action='store',
                      dest='mode',
                      choices=['new', 'existing', 'ready', 'tune'],
                      default=['new'],
                      help=help_str)

    parser.add_option('-i', '--img',
                      action='store',
                      dest='img_file',
                      help='input image file path',)

    parser.add_option('-k', '--colour_temp',
                      action='store',
                      dest='colour_temp',
                      default=3700,
                      help='colour temperature',)

    parser.add_option('-c', '--contrast',
                      action='store',
                      dest='contrast',
                      default=50,
                      help='colour contrast',)

    parser.add_option('-g', '--gamma',
                      action='store',
                      dest='gamma',
                      default=2.0,
                      help='gamma value',)

    parser.add_option('-n', '--runs',
                      action='store',
                      dest='runs',
                      default=1,
                      help='number of runs',)

    parser.add_option('-t', '--timer',
                      action='store_true',
                      dest='timer',
                      default=False,
                      help='True : report execution time, \
                            False: do not collect timing info',)

    parser.add_option('-d', '--display',
                      action='store_true',
                      dest='display',
                      default=False,
                      help='True : display output image, \
                            False: do not display output image',)

    parser.add_option('--cxx',
                      action='store',
                      dest='cxx',
                      choices=['g++', 'icpc'],
                      default=['g++'],
                      help='CXX Compiler')

    parser.add_option('--cxx_flags',
                      action='store',
                      dest='cxx_flags',
                      default=['-O3'],
                      help='CXX Compiler flags')

    parser.add_option('--graph-gen',
                      action='store_true',
                      dest='graph_gen',
                      default=False,
                      help='True : generate .dot file of pipeline graph, \
                            False: don\'t',)

    (options, args) = parser.parse_args()

    return options
