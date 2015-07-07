# Making things compatible for python 3
# Yet to figure out how to make range like xrange
from __future__ import absolute_import, division, print_function

from fractions import Fraction
import sys
sys.path.insert(0, '../../optimizer')
sys.path.insert(0, '../../frontend')

from Compiler import *
from Constructs import *

def test_harris_corner():

    print()
    print("test_harris")
    R = Parameter(Int, "R")
    C = Parameter(Int, "C")
    x = Variable(Int, "x")
    y = Variable(Int, "y")

    row = Interval(Int, 0, R+1)
    col = Interval(Int, 0, C+1)

    cond = Condition(x, '>=', 1) & Condition(x, '<=', R) & \
           Condition(y, '<=', C) & Condition(y, '>=', 1)

    condRed = Condition(x, '>=', 2) & Condition(x, '<=', R-1) & \
              Condition(y, '<=', C-1) & Condition(y, '>=', 2)


    img = Image(Float, "img", [R+2, C+2])

    Iy = Function(([x, y], [row, col]), Float, "Iy")
    Iy.defn = [ Case(cond, img(x-1, y-1)*(-1.0/12) + img(x-1, y+1)*(1.0/12) + \
                           img(x, y-1)*(-2.0/12) + img(x, y+1)*(2.0/12) + \
                           img(x+1, y-1)*(-1.0/12) + img(x+1, y+1)*(1.0/12)) ]

    Ix = Function(([x, y], [row, col]), Float, "Ix")
    Ix.defn = [ Case(cond, img(x-1, y-1)*(-1.0/12) + img(x+1, y-1)*(1.0/12) + \
                           img(x-1, y)*(-2.0/12) + img(x+1, y)*(2.0/12) + \
                           img(x-1, y+1)*(-1.0/12) + img(x+1, y+1)*(1.0/12)) ]

    Ixx = Function(([x, y], [row, col]), Float, "Ixx")
    Ixx.defn = [ Case(cond, Ix(x, y) * Ix(x, y)) ]

    Iyy = Function(([x, y], [row, col]), Float, "Iyy")
    Iyy.defn = [ Case(cond, Iy(x, y) * Iy(x, y)) ]

    Ixy = Function(([x, y], [row, col]), Float, "Ixy")
    Ixy.defn = [ Case(cond, Ix(x, y) * Iy(x, y)) ]

    Sxx = Function(([x, y], [row, col]), Float, "Sxx")
    Sxx.defn = [ Case(condRed, Ixx(x-1, y-1) + Ixx(x-1, y) + Ixx(x-1, y+1) +\
                               Ixx(x, y-1)   + Ixx(x, y)   + Ixx(x, y+1) +\
                               Ixx(x+1, y-1) + Ixx(x+1, y) + Ixx(x+1, y+1)) ]

    Syy = Function(([x, y], [row, col]), Float, "Syy")
    Syy.defn = [ Case(condRed, Iyy(x-1, y-1) + Iyy(x-1, y) + Iyy(x-1, y+1) +\
                               Iyy(x, y-1)   + Iyy(x, y)   + Iyy(x, y+1) +\
                               Iyy(x+1, y-1) + Iyy(x+1, y) + Iyy(x+1, y+1)) ]

    Sxy = Function(([x, y], [row, col]), Float, "Sxy")
    Sxy.defn = [ Case(condRed, Ixy(x-1, y-1) + Ixy(x-1, y) + Ixy(x-1, y+1) +\
                               Ixy(x, y-1)   + Ixy(x, y)   + Ixy(x, y+1) +\
                               Ixy(x+1, y-1) + Ixy(x+1, y) + Ixy(x+1, y+1)) ]

    det = Function(([x, y], [row, col]), Float, "det")
    det.defn = [ Case(condRed, Sxx(x, y) * Syy(x, y) - Sxy(x, y) * Sxy(x, y)) ]

    trace = Function(([x, y], [row, col]), Float, "trace")
    trace.defn = [ Case(condRed, Sxx(x, y) + Syy(x, y)) ]

    harris = Function(([x, y], [row, col]), Float, "harris")
    harris.defn = [ Case(condRed, 
                         det(x, y) - 0.04 * trace(x, y) * trace(x, y)) ]

    pipeline = buildPipeline([harris], grouping = [[Ix, Iy, Ixx, Iyy, Ixy]])

    filename = 'harris_graph.dot'
    pipeline.originalGraph.write(filename)
    
    filename = 'harris_graph_grouped.dot'
    g = pipeline.drawPipelineGraph()
    g.write(filename)

def test_downsample():

    print()
    print("test_downsample")
    R = Parameter(Int, "R")
    C = Parameter(Int, "C")
    x = Variable(Int, "x")
    y = Variable(Int, "y")

    L = 10

    row = {}
    col = {}
    cond = {}
    RL = {}
    CL = {}
    for l in range(0, L):
        RL[l] = (R/(1<<l))
        CL[l] = (C/(1<<l))

        row[l] = Interval(Int, 0, RL[l]+1)
        col[l] = Interval(Int, 0, CL[l]+1)

        cond[l] = Condition(x, '>=', 1) & Condition(x, '<=', RL[l]) & \
                  Condition(y, '>=', 1) & Condition(y, '<=', CL[l])

    img = Image(Float, "img", [R+2, C+2])

    down = {}

    for l in range(0, L):
        if l == 0:
            up = img
        else:
            up = down[l-1]

        down[l] = Function(([x, y], [row[l], col[l]]), 
                           Float, "down_"+str(l))
        down[l].defn = [ Case(cond[l], 
                             (up(2*x-1, 2*y-1) \
                           +  up(2*x+1, 2*y-1) \
                           +  up(2*x-1, 2*y+1) \
                           +  up(2*x+1, 2*y+1)) * 0.0625 \
                           + (up(2*x-1, 2*y  ) \
                           +  up(2*x+1, 2*y  ) \
                           +  up(2*x  , 2*y-1) \
                           +  up(2*x  , 2*y+1)) * 0.125 \
                           +  up(2*x  , 2*y  )  * 0.25) ]

    # manual grouping
    group1 = []
    for l in range(0, 3):
        group1.append(down[l])

    group2 = []
    for l in range(3, L):
        group2.append(down[l])

    groups = [group1, group2]

    # build the pipeline
    pipeline = buildPipeline([down[L-1]], grouping = groups)

    filename = 'down_graph.dot'
    pipeline.originalGraph.write(filename)
 
    filename = 'down_graph_grouped.dot'
    g = pipeline.drawPipelineGraph()
    g.write(filename)
