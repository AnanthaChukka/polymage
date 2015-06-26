# Making things compatible for python 3
# Yet to figure out how to make range like xrange
from __future__ import absolute_import, division, print_function

import sys
sys.path.insert(0, '../optimizer')
from Constructs import *
import Poly as opt

# More Python 3 vs 2 mojo
try:
    import queue
except ImportError:
    import Queue as queue

import pygraphviz as pgv

def getParentsFromCompObj(comp):
    refs = comp.getObjects(Reference)
    # Filter out self and image references 
    refs = [ ref for ref in refs if not ref.objectRef == comp and \
                                 not isinstance(ref.objectRef, Image) ]
    return list(set([ref.objectRef for ref in refs]))

def getCompObjsAndDependenceMaps(outputs):
    """
    Find all the compute objects required for the outputs and
    also builds parent and children maps for the compute objects
    """
    compObjs = []
    compObjsParents = {}
    compObjsChildren = {}
    q = queue.Queue()
    for compObj in outputs:
        q.put(compObj)
    while not q.empty():
        obj = q.get()
        parentObjs = getParentsFromCompObj(obj)
        if obj not in compObjs:
            compObjs.append(obj)
            compObjsParents[obj] = parentObjs
            for parobj in parentObjs:
                if parobj in compObjsChildren:
                    if obj not in compObjsChildren[parobj]:
                        compObjsChildren[parobj].append(obj)
                else:
                    compObjsChildren[parobj] = [obj]                        
            if len(parentObjs) != 0:
                for r in parentObjs:
                    q.put(r)

    for comp in compObjs:
        if comp not in compObjsParents:
            compObjsParents[comp] = []
        if comp not in compObjsChildren:
            compObjsChildren[comp] = []

    return compObjs, compObjsParents, compObjsChildren

class Group:
    """ 
        Group is a part of the pipeline which realizes a set of computation
        objects. Scheduling and storage allocation is done at the level of a 
        group. A group also maintains a polyhedral representation of the 
        computation objects when possible.
    """
    # Construct a group from a set of language functions / reductions
    def __init__(self, _ctx, _compObjs, _paramConstraints):
        # All the computation constructs in the language derive from the
        # Function class. Input images cannot be part of a group.
        for comp in _compObjs:
            assert(isinstance(comp, Function))
            assert(not isinstance(comp, Image))

        self._compObjs  = _compObjs
        self._polyrep = None
        refs = []

        for comp in self._compObjs:
            refs += comp.getObjects(Reference)

        self._inputs = list(set([ref.objectRef for ref in refs \
                            if isinstance(ref.objectRef, Image)]))

        # Create a polyhedral representation if possible
        polyhedral = True

        # Currently doing extraction only when all the computeObjs
        # domains are affine. This can be revisited later. 
        for comp in self._compObjs:
            if (not comp.hasBoundedIntegerDomain()):
                polyhedral = False
        if polyhedral:
            self._polyrep = opt.PolyRep(_ctx, self, _paramConstraints) 
        
    @property
    def computeObjs(self):
        return self._compObjs
    @property
    def polyRep(self):
        return self._polyrep
    @property
    def inputs(self):
        return self._inputs

    def getParameters(self):
        params = []
        for comp in self._compObjs:
            params = params + comp.getObjects(Parameter)
        return list(set(params))

    def isPolyhedral(self):
        polyhedral = True
        for comp in self._compObjs:
            if (not comp.hasBoundedIntegerDomain()):
                polyhedral = False
        return polyhedral

    def orderComputeObjs(self):
        # Order stores the numbering of each compute object 
        # when topologically sorted.
        order = {}
        # Initialize all the initial numbering to zero for
        # all compute objects in the group
        for comp in self._compObjs:
            order[comp] = 0
        # Doing a topological sort in an iterative fashion
        change = True
        while(change):
            change = False
            for comp in self._compObjs:
                parentObjs = getParentsFromCompObj(comp)
                for pobj in parentObjs:
                    if (pobj in order  and (order[pobj] >= order[comp])):
                        order[comp] = order[pobj] + 1
                        change = True
        return order
    
    def __str__(self):
        comp_str  = "\n\n".join([comp.__str__() for comp in self._compObjs]) + '\n'
        parent_str = "Parent Objects: " + \
                     ", ".join([ item.name for item in self._parentObjs]) + '\n' 
        return comp_str + '\n' + parent_str + '\n' + self._polyrep.__str__()

class Pipeline:
    def __init__(self, _ctx, _outputs, _paramConstraints):
        # Name of the pipleline is a concatenation of the names of the 
        # pipeline outputs.
        _name = ""
        for out in _outputs:
            _name = _name + out.name
        self._name   = _name
        
        self._ctx = _ctx
        self._orgOutputs = _outputs
        self._paramConstraints = _paramConstraints

        # Maps from a compute object to its parents and children
        self._orgCompObjs, _ , _ = \
                                getCompObjsAndDependenceMaps(self._orgOutputs)
        # TODO see if there is a cyclic dependency between the compute objects
        # self references are not treated as cycles

        # Clone the computation objects i.e. functions and reductions
        self._cloneMap = {}
        for comp in self._orgCompObjs:
            self._cloneMap[comp] = comp.clone()
        self._outputs = [ self._cloneMap[obj] for obj in self._orgOutputs]     

        # Modify the references in the cloned objects (which refer to 
        # the original objects)  
        for comp in self._orgCompObjs:
            refs = comp.getObjects(Reference)
            for ref in refs:
                if not isinstance(ref.objectRef, Image): 
                    ref._replaceRefObject(self._cloneMap[ref.objectRef])

        self._compObjs, self._compObjsParents, self._compObjsChildren = \
                                getCompObjsAndDependenceMaps(self._outputs)            

        # Create a group for each pipeline function / reduction and compute
        # maps for parent and child relations between the groups
        self._groups, self._groupParents, self._groupChildren = \
                                        self.buildInitialGroups(self._compObjs, 
                                                                self._compObjsParents,
                                                                self._compObjsChildren)

        # Store the initial pipeline graph. The compiler can modify 
        # the pipeline by inlining functions.
        self._initialGraph = self.drawPipelineGraph()

        # Make a list of all the input groups
        inputs = []
        for f in self._groups:
            inputs = inputs + self._groups[f].inputs

        self._inputs = list(set(inputs))

    @property
    def groups(self):
        return self._groups

    @property
    def name(self):
        return self._name

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def originalGraph(self):
        return self._initialGraph
    
    def buildInitialGroups(self, compObjs, compObjsParents, compObjsChildren):
        """
          Create a pipeline graph where the nodes are a group and the edges
          represent the dependences between the groups. The initial graph
          has each compute object in a separate group. 
        """
        # TODO correct the comment
        groupMap = {}
        groupParents = {}
        groupChildren = {}
        for comp in compObjs:
            groupMap[comp] = Group(self._ctx, [comp], self._paramConstraints)
        
        for comp in groupMap:
            groupParents[groupMap[comp]] = [ groupMap[p] for p in \
                                                        compObjsParents[comp] ]
            groupChildren[groupMap[comp]] = [ groupMap[c] for c in \
                                                        compObjsChildren[comp] ]
        return groupMap, groupParents, groupChildren    
    
    def getParameters(self):
        params=[]
        for group in self._groups.values():
            params = params + group.getParameters()
        return list(set(params))

    def drawPipelineGraph(self):
        G = pgv.AGraph(strict=False, directed=True)
        groupList = list(set([self._groups[f] for f in self._groups]))
        for obj in self._compObjs:
            for pobj in self._compObjsParents[obj]:
                G.add_edge(pobj.name, obj.name)
        # TODO add input nodes to the graph 
        for i in range(0, len(groupList)):
            subGraphNodes = []
            for obj in self._compObjs:
                if self._groups[obj] == groupList[i]:
                    subGraphNodes.append(obj.name)
            G.add_nodes_from(subGraphNodes)
            G.add_subgraph(nbunch = subGraphNodes, 
                           name = "cluster_" + str(i))

        G.layout(prog='dot')
        return G

    def boundsCheckPass(self):
        """ 
            Bounds check pass analyzes if function values used in the compute
            objects are within the domain of the functions. Static analysis is
            only possible when the references to function values are regular
            i.e. they are not data dependent. We restrict ourselves to affine
            references.
        """
        for group in self._groups.values():
            for child in group.childGroups:
                opt.checkRefs(child, group)              
            for inp in group.inputs:
                # Creating a computation group for an input which is given
                # is meaningless. Ideally it should be done in a clean way
                # currently abusing group for construction of a polyhedral
                # representation
                inpGroup = Group([inp], self._ctx, self._paramConstraints,
                                 self._paramEstimates, self._tileSizes,
                                 self._sizeThreshold, self._groupSize,
                                 self._outputs)
                opt.checkRefs(group, inpGroup)

    def inlinePass(self):
        """ 
            Inline pass takes all the inlining decisions and inlines functions 
            at their use points in other groups.
        """
        # Guidelines for inlining
        # -- Inlining functions which have multiple case constructs can quickly 
        #    turn out to be messy code generation nightmare.
        # -- Splitting an use site can occur when the inlined function is defined 
        #    by different expression in different parts of the function domain. 
        #    Functions which are defined by a single expression over the entire 
        #    domain are good candidates for inlining.
        #    Example :
        #    f(x) = g(x-1) + g(x) + g(x+1) when (1 <= x <= N-1)
        #         = 0                      otherwise
        #    h(x) = f(x-1) + f(x) + f(x+1) when (1 <= x <= N-1)
        #         = 0                      otherwise
        #    Inlining f into h will result in splitting as shown below
        #    h(x) = g(x-2) + g(x-1) + g(x) +      when (2 <= x <= N-2) 
        #           g(x-1) + g(x) + g(x+1) +
        #           g(x)   + g(x+1) + g(x+2)
        #         = 2*g(x-1) + 3*g(x) + 2*g(x+1)  when (x == 1) 
        #           + g(3)      
        #         = g(x-2) + 2*g(x-1) +           when (x == N-1)
        #           3*g(x) + 2*g(x+1)
        #         = 0                             otherwise
        #    For multiple dimensions and complex expression it gets ugly fast
        # -- Inlining without splitting is possible even when the function is 
        #    defined by mutliple expressions. When the function references at
        #    the use site can all be proved to be generated from a single 
        #    expression.
        #    Example :
        #    f(x) = g(x-1) + g(x) + g(x+1) when (1 <= x <= N-1)
        #         = 0                      otherwise
        #    h(x) = f(x-1) + f(x) + f(x+1) when (2 <= x <= N-2)
        #         = 0                      otherwise
        #    Inlining f into h will not result in splitting as shown below
        #    h(x) = g(x-2) + g(x-1) + g(x) +      when (2 <= x <= N-2) 
        #           g(x-1) + g(x) + g(x+1) +
        #           g(x)   + g(x+1) + g(x+2)
        #         = 0                             otherwise
        # -- Users should be enouraged to write functions in a form that makes
        #    inlining easy.
        inlinedCompObjs = []
        for directive in self._inlineDirectives:
            # Only function constructs can be inlined for now
            assert isinstance(directive, Function)
            # Does inling into a fused group cause problems?
            parentGroup = self._groups[directive]
            assert parentGroup.computeObjs[0] not in self._outputs
            for child in parentGroup.childGroups:
                refToInlineExprMap = opt.inline(child, parentGroup, noSplit = True)
                child.computeObjs[0].inlineRefs(refToInlineExprMap)
            # Recompute group graph
            self._groups = self.buildGroupGraph()

    def getOrderedGroups(self):
        # Topological sorting of groups.
        # Quick and dirty way of doing things have to revisit
        groupOrder = {}
        groupList = [ (len(s.parentGroups), s) for s in self._groups.values()]
        level = 0
        while len(groupList) > 0:
            levelAssigned = [ s for s in groupList if s[0] == 0]
            for assigned in levelAssigned:
                groupOrder[assigned[1]] = level
                groupList.remove(assigned)
                def decChild(s):
                    if s[1] in assigned[1].childGroups:
                        return (s[0]-1, s[1])
                    else: 
                        return s
                groupList = [ decChild(s) for s in groupList ] 
            level = level + 1
        return sorted(groupOrder.items(), key=lambda s: s[1])
  
    # TODO printing the pipeline 
    def __str__(self):
        return_str = "Final Group: " + self._name + "\n"
        for s in self._groups:
            return_str = return_str + s.__str__() + "\n"
        return return_str
