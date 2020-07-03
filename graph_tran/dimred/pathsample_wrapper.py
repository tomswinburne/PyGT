r""" 
Wrapper to PATHSAMPLE implementation of graph transformation
------------------------------------------------------------
This module provides a Python interface for the graph transformation
implementation in the `PATHSAMPLE program <http://www-wales.ch.cam.ac.uk/PATHSAMPLE/>`_
available under the GNU General Public License. In addition to constructing
a discrete-state network-representation of the energy landscape, the
PATHSAMPLE program provides tools to compute rates and mean
first passage times between product and reactant regions on the landscape.

The program also includes a REGROUPFREE routine, which groups together
potential energy or free energy minimuma separated by low-lying barriers
within a free energy regrouping threshold. This routine can be used to
pre-process a network by lumping together rapidly interconverting states.

The implementation of graph transformation in Fortran is preferrable
for large networks on the order of 100,000 nodes. For smaller networks, the python
implementation available in `graph_tran.gt_tools` should suffice.

* Used the ``ParsedPathsample`` class to read in the initial `pathdata` file, which specifies the input keyworks
  for the PATHSAMPLE program. Modify the input parameters as desired using this class.
* Use the ``ScanPathsample`` class to run calculations with PATHSAMPLE and parse output files
  to obtain the desired quantities (i.e. rate constants, mean first passage times, groups identified by REGROUPFREE, etc.)

"""

import re
import numpy as np
import os
import glob
import pandas as pd
from pathlib import Path
import subprocess

INDEX_OF_KEYWORD_VALUE = 15
#path to bin for compiled programs
PATHSAMPLE = "/home/dk588/svn/PATHSAMPLE/build/gfortran/PATHSAMPLE"
disconnectionDPS = "/home/dk588/svn/DISCONNECT/source/disconnectionDPS"

def write_communities(self, communities, commdat):
    """ Write a single-column file `commdat` where each line is the
    community ID (zero-indexed) of the minima given by the line
    number.

    Parameters
    ----------
    communities : dict
        mapping from minima ID (1-indexed) to community ID (1-indexed)
    commdat : .dat file name
        full path to file to which to write communitiy assignments (0-indexed)

    """
    commdat = Path(commdat)
    if commdat.exists():
        raise ValueError(f'The file {commdat} already exists. Write to a new file')
    nnodes = len(communities) #number of keys = number of minima
    with open(commdat, 'w') as f:
        for min in range(1, nnodes+1):
            f.write(f'{np.array(communities[min]) - 1}\n')

class ParsedPathsample(object):
    """ Class to parse input keywords and output files
    utilized by the PATHSAMPLE program. To initialize a ``ParsedPathsample`` object,
    supply the constructor with the `pathdata` argument, which should include a full path
    to the pathdata file.

    Attributes
    ----------
    path : Path object
        path to directory containing all relevant input files, including `pathdata` file, which specifies input keywords.
    input : dict
        dictionary containing PATHSAMPLE keywords and their values
    output : dict
        dictionary mapping observables to their values parsed from PATHSAMPLE output files.
    numInA : int
        Number of minima in A.
    numInB : int
        Number of minima in B.
    minA : list
        IDs of minima in A set (0-indexed)
    minB : list
        IDs of minima in B set (0-indexed)

    """

    def __init__(self, pathdata, outfile=None):
        self.output = {} #dictionary of output (i.e. temperature, rates)
        self.input = {} #dictionary of PATHSAMPLE keywords
        self.numInA = 0 #number of minima in A
        self.numInB = 0 #number of minima in B
        self.minA = [] #IDs of minima in A set (0-indexed)
        self.minB = [] #IDs of minima in B set (0-indexed)
        #read in numInA, numInB, minA, minB from min.A and min.B files
        self.path = Path(pathdata).parent.absolute()
        self.parse_minA_and_minB(self.path/'min.A', self.path/'min.B')
        if outfile is not None:
            self.parse_output(outfile)
        self.parse_input(pathdata)

    def parse_minA_and_minB(self, minA, minB):
        """Read in the number of minima and the minima IDs in the A and B sets
        from the min.A and min.B files and set the self.minA, self.numInA,
        self.minB and self.numInB attributes. Note, minima IDs in these files
        correspond to line numbers in min.data which are (1-indexed).

        Parameters
        ----------
        minA : str or Path object
            full path to min.A file
        minB : str or Path object
            full path to min.B file
    
        """
        Aids = []
        with open(minA) as f:
            for line in f:
                Aids = Aids + line.split()
        Aids = np.array(Aids).astype(int)
        maxinA = Aids[0]
        #adjust indices by 1 cuz python is 0-indexed
        Aids = Aids[1:] - 1
        self.minA = Aids
        self.numInA = maxinA
        #and for min.B
        Bids = []
        with open(minB) as f:
            for line in f:
                Bids = Bids + line.split()
        Bids = np.array(Bids).astype(int)
        maxinB = Bids[0]
        Bids = Bids[1:] - 1
        self.minB = Bids
        self.numInB = maxinB

    def sort_A_and_B(self, minA, minB, mindata):
        """Sort the minima in min.A and min.B according to their energies."""
        self.parse_minA_and_minB(minA, minB)
        min_nrgs = np.loadtxt(mindata).astype(float)[:,0]
        minA_nrgs = min_nrgs[self.minA]
        sorted_idAs = np.argsort(minA_nrgs)
        self.minA = self.minA[sorted_idAs]
        minB_nrgs = min_nrgs[self.minB]
        sorted_idBs = np.argsort(minB_nrgs)
        self.minB = self.minB[sorted_idBs]
        self.write_minA_minB(self.path/'min.A', self.path/'min.B')

    def define_A_and_B(self, numInA, numInB, sorted=True, mindata=None):
        """Define an A and B set as a function of the number of minima in A
        and B based on their energies. """
        if sorted:
            #min.A and min.B are already sorted by energy
            #just change numInA and numInB
            self.numInA = numInA
            self.numInB = numInB
            return
        if mindata is None:
            mindata = self.path/'min.data'
        #first column of min.data gives free energies
        min_nrgs = np.loadtxt(mindata).astype(float)[:,0]
        #index minima by 0 to correspond to indices of min_nrgs
        minIDs = np.arange(0, len(min_nrgs), 1)
        minA_nrgs = min_nrgs[self.minA]
        minB_nrgs = min_nrgs[self.minB]
        idA = np.argpartition(minA_nrgs, numInA)
        self.minA = self.minA[idA[:numInA]]
        self.numInA = numInA
        idB = np.argpartition(minB_nrgs, numInB)
        self.minB = self.minB[idB[:numInB]]
        self.numInB = numInB

    def write_minA_minB(self, minA, minB):
        """Write a min.A and min.B file based on minIDs
        specified in self.minA and self.minB."""
        with open(minA,'w') as f:
            f.write(str(self.numInA)+'\n') #first line is number of minima
            for min in self.minA:
                f.write(str(min+1)+'\n')

        with open(minB,'w') as f:
            f.write(str(self.numInB)+'\n') #first line is number of minima
            for min in self.minB:
                f.write(str(min+1)+'\n')

    def parse_output(self, outfile):
        """Parses PATHSAMPLE output for key quantities printed out by the
        NGT and WAITPDF subroutines, namely Temperature, kSS, kNSS, kF, and MFPTs
        for the A->B and B->A directions. Stores quantities in self.output.
        """
        with open(outfile) as f:
            for line in f:
                if not line:
                    continue
                words = line.split()
                if len(words) > 1:
                    if words[0] == 'Temperature=':
                        #parse and spit out self.temperature
                        self.output['temperature'] = float(words[1])
                    if words[0] == 'NGT>' and words[1]=='kSS':
                        self.output['kSSAB'] = float(words[3])
                        self.output['kSSBA'] = float(words[6])
                    if words[0] == 'NGT>' and words[1]=='kNSS(A<-B)=':
                        self.output['kNSSAB'] = float(words[2])
                        self.output['kNSSBA'] = float(words[4])
                    if words[0] == 'NGT>' and words[1]=='k(A<-B)=':
                        self.output['kAB'] = float(words[2])
                        self.output['kBA'] = float(words[4])
                        self.output['MFPTAB'] = float(words[6])
                        self.output['MFPTBA'] = float(words[8])
                    if words[0] == 'waitpdf>' and words[2]=='A<-B':
                        self.output['tau*AB'] = float(words[7])
                    if words[0] == 'waitpdf>' and words[2]=='B<-A':
                        self.output['tau*BA'] = float(words[7])
                        #this is the last line we're interested in
                        break

    def parse_GT_intermediates(self, outfile):
        """Returns a pandas DataFrame of GT quantities, including branching
        probabilities and MFPTs from individual source nodes, after disconnection of sources.
        """
        raise NotImplementedError('This function has not been implemented yet.')

        regex1 = re.compile('NGT> for (A|B) minimum\s+([0-9]+) ' +
                            '(?:P_{Ba}|P_{Ab})=\s+([0-9.E-]+) and time ' +
                            'tau\^F_(?:a|b)=\s+([0-9.E-]+)')

        regex2 = re.compile('NGT> for (A|B) minimum\s+([0-9]+) ' +
                            '(?:tau\^F_a/P_{Ba} = T_{Ba}, p_a/P_A, '+
                            'weight/T_{Ba}|tau\^F_b/P_{Ab} = T_{Ab}, p_b/P_B, ' +
                            'weight/T_{Ab})=\s+([0-9.E-]+)\s+([0-9.E-]+)\s+([0-9.E-]+)')

        dfs = []
        finished = False
        with open(outfile) as f:
            olddf = pd.DataFrame(columns=['JsetLabel','min','PIj','tauF_j'])
            olddf2 = pd.DataFrame(columns=['T_Ij','pihat_j'])
            for line in f:
                if not line:
                    continue
                match = regex1.match(line)
                if match is not None:
                    label, min, PIj, tau_j = match.groups()
                    df = pd.DataFrame()
                    df['JsetLabel'] = [label]
                    df['min'] = [min]
                    df['PIj'] = [PIj]
                    df['tauF_j'] = [tau_j]
                    olddf = pd.concat([olddf, df], axis=0, ignore_index=True)
                    continue
                match2 = regex2.match(line)
                if match2 is not None:
                    label, min, T_Ij, pihat_j, pihat_over_TIj = match2.groups()
                    df2 = pd.DataFrame()
                    df2['T_Ij'] = [T_Ij]
                    df2['pihat_j'] = [pihat_j]
                    olddf2 = pd.concat([olddf2, df2], axis=0,
                                       ignore_index=True)
            bigdf = pd.concat([olddf, olddf2], axis=1, ignore_index=True)
            bigdf.to_csv('csvs/GT_quantities.csv')


    def parse_dumpgroups(self, mingroupsfile, grouptomin=False):
        """Parse the `minima_groups.{temp}` file outputted by the DUMPGROUPS
        keyword in PATHSAMPLE. Returns a dictionary mapping minID to groupID
        (if grouptomin is False), or a dirctionary mapping groupID to minID (if
        grouptomin is True).
        Both IDs are 1-indexed."""

        communities = {}
        with open(mingroupsfile) as f:
            group = []
            for line in f:
                words = line.split()
                if len(words) < 1:
                    continue
                if words[0] != 'group':
                    group += words
                else: #reached end of group
                    groupid = int(words[1])
                    #update dictionary with each min's group id
                    if grouptomin: #groupid --> [list of min in group]
                        communities[groupid] = [int(min) for min in group]
                    else: #minid --> groupid
                        for min in group:
                            communities[int(min)] = groupid
                    group = [] #reset for next group

        return communities

    def draw_disconnectivity_graph_AB(self, value, temp):
        """Draw a Disconnectivity Graph colored by the minima in A and B after
        running REGROUPFREE at threshold `value` and temperature `temp`."""

        #extract group assignments for this temperature/Gthresh
        communities = self.parse_dumpgroups(self.path/f'minima_groups.{temp:.10f}',
                                    grouptomin=True)
        self.parse_minA_and_minB(self.path/f'min.A.regrouped.{temp:.10f}',
                                    self.path/f'min.B.regrouped.{temp:.10f}')
        #calculate the total number of minima in A and B
        regroupedA = []
        for a in self.minA: #0-indexed so add 1
            #count the number of minima in that group, increment total count
            regroupedA += communities[a+1]
        sizeOfA = len(regroupedA)
        regroupedB = []
        for b in self.minB: #0-indexed so add 1
            #count the number of minima in that group, increment total count
            regroupedB += communities[b+1]
        sizeOfB = len(regroupedB)
        #write minima in A group to file for TRMIN
        with open(self.path/f'minA.{value:.2f}.T{temp:.2f}.dat', 'w') as fi:
            for min in regroupedA:
                fi.write(f'{min}\n')
        #write minima in B group to file for TRMIN
        with open(self.path/f'minB.{value:.2f}.T{temp:.2f}.dat', 'w') as fi:
            for min in regroupedB:
                fi.write(f'{min}\n')
        #modify dinfo file to include above files
        #TODO: make sure min.data, ts.data specified.
        #TODO: make sure TRMIN already specified in existing dinfo file
        os.system(f"mv {self.path/'dinfo'} {self.path/'dinfo.original'}")
        #create a copy of the dinfo file
        with open(self.path/'dinfo', 'w') as newdinfo:
            with open(self.path/'dinfo.original','r') as ogdinfo:
                #copy over all lines from previous dinfo file except TRMIN
                for line in ogdinfo:
                    words = line.split()
                    print(words)
                    #change only the TRMIN line
                    if words[0]=='TRMIN':
                        newdinfo.write(f'TRMIN 2 998 minA.{value:.2f}.T{temp:.2f}.dat ' +
                                f'minB.{value:.2f}.T{temp:.2f}.dat\n')
                    else:
                        newdinfo.write(line)
        #run disconnectionDPS
        os.system(f"{disconnectionDPS}")
        os.system("evince tree.ps")

    def parse_input(self, pathdatafile):
        """Store keywords in `pathdata` file as a dictionary
        mapping to values, which are read in as strings.
        """
        name_value_re = re.compile('([_A-Za-z0-9][_A-Za-z0-9]*)\s*(.*)\s*')

        with open(pathdatafile) as f:
            for line in f:
                if not line or line[0]=='!':
                    continue
                match = name_value_re.match(line)
                if match is None:
                    continue
                name, value = match.groups()
                self.input.update({name: value})

    def write_input(self, pathdatafile):
        """Writes out a valid pathdata file based on keywords
        in self.input"""
        with open(pathdatafile, 'w') as f:
            #first 3 lines are garbage
            #f.write('! PATHSAMPLE input file generated from\n')
            #f.write('! ParsedPathsample class\n\n')
            for name in self.input:
                name_length = len(name)
                #the value of the keyword begins on the 15th index of the line
                #add the appropriate number of spaces before the value
                numspaces = INDEX_OF_KEYWORD_VALUE - name_length
                f.write(str(name).upper() + ' '*numspaces + str(self.input[name]) + '\n')

    def append_input(self, name, value):
        """Add a new keyword or update an existing keyword to the PATHSAMPLE input."""
        self.input.update({name: value})

    def comment_input(self, name):
        """Delete a keyword from pathdata file."""
        self.input.pop(name, None)


class ScanPathsample(object):
    """ Interface to run the REGROUPFREE, DUMPINFOMAP, NGT, and WAITPDF
    subroutines of the PATHSAMPLE program. Uses the ``ParsedPathsample``
    class to control input parameters and and parse output.
    
    Attributes
    ----------
    pathdatafile : str or Path object
        full path to `pathdata` file, which stores input keywords for the PATHSAMPLE program.
    parse : ParsedPathsample object
        stores input and output of PATHSAMPLE runs.
    path : Path object
        full path to directory containing all input and output files.
    suffix : str
        suffix of `rates_{suffix}.csv` file. Defaults to ''
    outbase : str
        prefix for PATHSAMPLE output files. Defaults to 'out'

    """


    def __init__(self, pathdata, outbase=None, suffix=''):
        self.pathdatafile = pathdata #base input file to modify
        self.parse = ParsedPathsample(pathdata=pathdata)
        self.path = Path(pathdata).parent.absolute()
        self.suffix = suffix #suffix of rates_{suffix}.csv file output
        self.outbase = outbase #prefix for pathsample output files
        if outbase is None:
            self.outbase = 'out'

    def get_intermicrostate_mfpts_GT_PATHSAMPLE(temp, n=None):
        """Use PATHSAMPLE to compute the matrix of inter-microstate MFPTs
        between all pairs of the `n` nodes at the specified temperature `temp`.

        Parameters
        ----------
        temp : float
            Temperature at which to run GT calculations.
        n : int
            Number of nodes.
        
        Returns
        -------
        mfpt : np.ndarray[float64] (n,n)
            Matrix of inter-minima MFPTs.

        """

        files_to_modify = [self.path/'min.A', self.path/'min.B']
        if n is None:
            mindata = np.loadtxt(self.path/'min.data')
            n = mindata.shape[0]
        for f in files_to_modify:
            if not f.exists():
                print(f'File {f} does not exists')
                raise FileNotFoundError 
            os.system(f'mv {f} {f}.original')
        
        parse = self.parse
        parse.append_input('NGT', '0 T')
        parse.comment_input('WAITPDF')
        parse.append_input('TEMPERATURE', f'{temp}')
        parse.write_input(self.path/'pathdata')
        mfpt = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i < j:
                    parse.minA = [i] 
                    parse.numInA = 1
                    parse.minB = [j]
                    parse.numInB = 1
                    parse.write_minA_minB(self.path/'min.A', self.path/'min.B')
                    #run PATHSAMPLE
                    outfile_name = self.path/f'out.{i+1}.{j+1}.T{temp}'
                    outfile = open(outfile_name, 'w')
                    subprocess.run(f"{PATHSAMPLE}", stderr=subprocess.STDOUT, stdout=outfile, cwd=self.path)
                    #parse output
                    parse.parse_output(outfile=self.path/f'out.{i+1}.{j+1}.T{temp}')
                    mfpt[i, j] = parse.output['MFPTAB']
                    mfpt[j, i] = parse.output['MFPTBA']
                    Path(outfile_name).unlink()
                    print(f'Calculated MFPT for states ({i}, {j})')

        #restore original min.A and min.B files
        for f in files_to_modify:
            os.system(f'mv {f}.original {f}')

        return mfpt

    def get_MFPTAB_PATHSAMPLE(self, A, B, communities, temp, n):
        """Run a single GT calculation using the READRATES keyword in PATHSAMPLE
        to get the A<->B mean first passage time at specified temperature.
        
        Parameters
        ----------
        A : int
            community ID (0-indexed) of A set.
        B : int
            community ID (0-indexed) of B set.
        communities : dict
            dictionary mapping community IDs (1-indexed) to node IDs (1-indexed).
        temp : float
            temperature at which to run PATHSAMPLE calculations.
        n : int
            number of nodes, needed for READRATES keyword.

        Returns
        -------
        MFPTAB : float
            Mean first passage time A <- B
        MFPTBA : float
            Mean first passage time B <- A

        """

        parse = self.parse
        files_to_modify = [self.path/'min.A', self.path/'min.B']
        for f in files_to_modify:
            if not f.exists():
                print(f'File {f} does not exists')
                raise FileNotFoundError 
            os.system(f'mv {f} {f}.original')
        #update min.A and min.B with nodes in A and B
        parse.minA = np.array(communities[A+1]) - 1
        parse.numInA = len(communities[A+1])
        parse.minB = np.array(communities[B+1]) - 1
        parse.numInB = len(communities[B+1])
        parse.write_minA_minB(self.path/'min.A', self.path/'min.B')
        parse.append_input('NGT', '0 T')
        parse.append_input('TEMPERATURE', f'{temp}')
        parse.append_input('READRATES', f'{n}')
        parse.write_input(self.path/'pathdata')
        #run PATHSAMPLE
        outfile = open(self.path/f'out.{A+1}.{B+1}.T{temp}', 'w')
        subprocess.run(f"{PATHSAMPLE}", stdout=outfile, cwd=self.path)
        #parse output
        parse.parse_output(outfile=self.path/f'out.{A+1}.{B+1}.T{temp}')
        return parse.output['MFPTAB'], parse.output['MFPTBA']


    def calc_inter_community_rates_GT(self, C1, C2, communities):
        """Calculate k_{C1<-C2} using GT. Here, C1 and C2 are community IDs
        (i.e. groups identified in DUMPGROUPS file from REGROUPFREE). This
        function isolates the minima in C1 union C2 and the transition states
        that connect them and feeds this subnetwork into PATHSAMPLE, using the
        NGT keyword to calculate inter-community rates."""

        #minima to isolate
        mintoisolate = communities[C1] + communities[C2]
        #parse min.data and write a new min.data file with isolated minima
        #also keep track of the new minIDs based on line numbers in new file
        newmin = {}
        j = 1
        with open(self.path/f'min.data.{C1}.{C2}', 'w') as newmindata:
            with open(self.path/'min.data','r') as ogmindata:
                #read min.data and check if line number is in C1 U C2
                for i, line in enumerate(ogmindata, 1):
                    if i in mintoisolate:
                        #save mapping from old minIDs to new minIDs
                        newmin[i] = j
                        #NOTE: these min have new line numbers now
                        #so will have to re-number min.A,min.B,ts.data
                        newmindata.write(line)
                        j += 1
                    
        #exclude transition states in ts.data that connect minima not in C1/2
        ogtsdata = pd.read_csv(self.path/'ts.data', sep='\s+', header=None,
                               names=['nrg','fvibts','pointgroup','min1','min2','itx','ity','itz'])
        newtsdata = []
        noconnections = True #flag for whether C1 and C2 are disconnected
        for ind, row in ogtsdata.iterrows():
            min1 = int(row['min1'])
            min2 = int(row['min2'])
            if min1 in mintoisolate and min2 in mintoisolate:
                # turn off noconnections flag as soon as one TS between C1 and
                # C2 is found
                if ((min1 in communities[C1] and min2 in communities[C2]) or
                (min1 in communities[C2] and min2 in communities[C1])):
                    noconnections = False
                #copy line to new ts.data file, renumber min
                modifiedrow = pd.DataFrame(row).transpose()
                modifiedrow['min1'] = newmin[min1]
                modifiedrow['min2'] = newmin[min2]
                modifiedrow['pointgroup'] = int(modifiedrow['pointgroup'])
                newtsdata.append(modifiedrow)
        if noconnections or len(newtsdata)==0:
            #no transition states between these minima, return 0
            print(f"No transition states exist between communities {C1} and {C2}")
            return 0.0, 0.0
        newtsdata = pd.concat(newtsdata)
        #write new ts.data file
        newtsdata.to_csv(self.path/f'ts.data.{C1}.{C2}',header=False, index=False, sep=' ')
        #write new min.A/min.B files with nodes in C1 and C2 (using new
        #minIDs of course)
        numInC1 = len(communities[C1])
        minInC1 = []
        for min in communities[C1]:
            minInC1.append(newmin[min] - 1)
        numInC2 = len(communities[C2])
        minInC2 = []
        for j in communities[C2]:
            minInC2.append(newmin[j] - 1)
        self.parse.minA = minInC1
        self.parse.minB = minInC2
        self.parse.numInA = numInC1
        self.parse.numInB = numInC2
        self.parse.write_minA_minB(self.path/f'min.A.{C1}', self.path/f'min.B.{C2}')
        #run PATHSAMPLE
        files_to_modify = [self.path/'min.A', self.path/'min.B',
                           self.path/'min.data', self.path/'ts.data']
        for f in files_to_modify:
            os.system(f'mv {f} {f}.old')
        os.system(f"cp {self.path}/min.A.{C1} {self.path}/min.A")
        os.system(f"cp {self.path}/min.B.{C2} {self.path}/min.B")
        os.system(f"cp {self.path}/min.data.{C1}.{C2} {self.path}/min.data")
        os.system(f"cp {self.path}/ts.data.{C1}.{C2} {self.path}/ts.data")
        outfile = open(self.path/f'out.{C1}.{C2}.T{temp}','w')
        subprocess.run(f"{PATHSAMPLE}", stdout=outfile, cwd=self.path)
        #parse output
        self.parse.parse_output(outfile=self.path/f'out.{C1}.{C2}.T{temp}')
        for f in files_to_modify:
            os.system(f'mv {f}.old {f}')
        #return rates k(C1<-C2), k(C2<-C1)
        return self.parse.output['kAB'], self.parse.output['kBA']

    def get_intercommunity_MFPTs_GT_PATHSAMPLE(self, temp, communities):
        """Use PATHSAMPLE to compute the matrix of true MFPTs between communities.

        Parameters
        ----------
        temp : float
            Temperature at which to calculate inter-community MFPTs.
        communities: dict
            dictionary mapping community IDs (1-indexed) to node IDs (1-indexed).

        Returns
        -------
        MFPT : np.ndarray[float64] (ncomms, ncomms)
            matrix of inter-community MFPTs computed with PATHSAMPLE

        """

        parse = self.parse
        files_to_modify = [self.path/'min.A', self.path/'min.B']
        for f in files_to_modify:
            if not f.exists():
                print(f'File {f} does not exists')
                raise FileNotFoundError 
            os.system(f'mv {f} {f}.original')
        N = len(communities)
        MFPT = np.zeros((N,N))
        for ci in range(N):
            for cj in range(N):
                if ci < cj:
                    parse.minA = np.array(communities[ci+1]) - 1
                    parse.numInA = len(communities[ci+1])
                    parse.minB = np.array(communities[cj+1]) - 1
                    parse.numInB = len(communities[cj+1])
                    parse.write_minA_minB(self.path/'min.A', self.path/'min.B')
                    #os.system(f'cat {self.path}/min.A')
                    #os.system(f'cat {self.path}/min.B')
                    parse.append_input('NGT', '0 T')
                    parse.append_input('TEMPERATURE', f'{temp}')
                    parse.write_input(self.path/'pathdata')
                    #run PATHSAMPLE
                    outfile = open(self.path/f'out.{ci+1}.{cj+1}.T{temp}', 'w')
                    subprocess.run(f"{PATHSAMPLE}", stdout=outfile, cwd=self.path)
                    #parse output
                    parse.parse_output(outfile=self.path/f'out.{ci+1}.{cj+1}.T{temp}')
                    MFPT[ci, cj] = parse.output['MFPTAB']
                    MFPT[cj, ci] = parse.output['MFPTBA']

        #restore original min.A and min.B files
        for f in files_to_modify:
            os.system(f'mv {f}.original {f}')

        return MFPT

    def dump_rates_full_network(self, temp=None):
        """ Dump rate matrix and stationary probabilities calculated in the
        PATHSAMPLE setup using the DUMP_INFOMAP keyword. Writes out `ts_conns.dat`,
        `ts_weights.dat`, and `stat_prob.dat` files.

        Note
        ----
        Make sure that any existing `stat_prob.dat`, `ts_conns.dat`, and `ts_weights.dat` files
        are deleted from the `self.path` directory.
        """
        files_to_write = [self.path/'ts_weights.dat',
                          self.path/'stat_prob.dat']
        #rename any pre-exisiting ts_weights.dat and stat_prob.dat files
        for f in files_to_write:
            if f.exists():
                os.system(f'mv {f} {f}.old')
        #remove any unnecessary keywords
        self.parse.comment_input('REGROUPFREE')
        self.parse.comment_input('DUMPGROUPS')
        self.parse.comment_input('NGT')
        #call dump_infomap to obtain Daniel's files
        self.parse.append_input('DUMPINFOMAP', '')
        if temp is None:
            temp = float(self.parse.input['TEMPERATURE'])
        else:
            self.parse.append_input('TEMPERATURE', f'{temp}')
        self.parse.write_input(self.pathdatafile)
        outfile = open(self.path/f'out.dumpinfo.{temp:.2f}','w')
        subprocess.run(f"{PATHSAMPLE}", stdout=outfile, cwd=self.path)
        #also write a ts_weights.dat file using info from ts.data
        tsdata = np.loadtxt(self.path/'ts.data')
        tsdata = tsdata[:,[3,4]].astype('int') #isolating minima ts connects
        np.savetxt(self.path/f'ts_conns_T{temp:.2f}.dat', tsdata, fmt='%d')
        #rename the files something useful
        for f in files_to_write:
            os.system(f'mv {f} {self.path}/{f.stem}_T{temp:.2f}.dat')

    def dump_rates_from_local_equilibrium_approx(self, temp):
        """ Dump rate matrix and stationary probabilities of the coarse-grained
        network resulting from the REGROUPFREE routine run at temperature `temp`. 
        Writes out `ts_conns.dat`, `ts_weights.dat`, and `stat_prob.dat` files 
        representing the regrouped network.
        """

        files_to_write = [self.path/'ts_weights.dat',
                          self.path/'stat_prob.dat']
        #rename any pre-exisiting ts_weights.dat and stat_prob.dat files
        for f in files_to_write:
            if f.exists():
                os.system(f'mv {f} {f}.old')
        #Rename regrouped files to min.A and min.B to pass as input to PATHSAMPLE
        files_to_modify = [self.path/'min.A', self.path/'min.B',
                           self.path/'min.data', self.path/'ts.data']
        for f in files_to_modify:
            os.system(f"mv {f} {f}.original")
            os.system(f"mv {f}.regrouped.{temp:.10f} {f}")
        # not interested in running NGT or REGROUPFREE, just want
        # PATHSAMPLE to dump kij rates and pi_i from network
        self.parse.comment_input('REGROUPFREE')
        self.parse.comment_input('DUMPGROUPS')
        self.parse.comment_input('NGT')
        self.parse.append_input('DUMPINFOMAP', '') #to obtain Daniel's files
        self.parse.write_input(self.pathdatafile)
        outfile = open(self.path/f'out.dumpinfo.LEA.{temp:.2f}','w')
        subprocess.run(f"{PATHSAMPLE}", stdout=outfile, cwd=self.path)
        #rename the files something useful
        for f in files_to_write:
            os.system(f'mv {f} {self.path}/{f.stem}_LEA_T{temp:.2f}.dat')
        #also write a ts_weights.dat file using info from ts.data
        tsdata = np.loadtxt(self.path/'ts.data')
        tsdata = tsdata[:,[3,4]].astype('int') #isolating minima ts connects
        np.savetxt(self.path/f'ts_conns_LEA_T{temp:.2f}.dat', tsdata, fmt='%d')

        #return files to original names
        for f in files_to_modify:
            os.system(f"mv {f} {f}.regrouped.{temp:.10f}")
            os.system(f"mv {f}.original {f}")

    def run_NGT_regrouped(self, Gthresh, temp):
        """After running REGROUPFREE at threshold `Gthresh` and temperature `temp`, 
        calculate rates using NGT on regrouped minima.
        
        Parameters
        ----------
        Gthresh : float
            Free energy regrouping threshold used by REGROUPFREE.
        temp : float
            Temperature.

        Returns
        -------
        rates : dict
            dictionary of NGT output quantities, including MFPT, kF, kSS, and kNSS 
            for the A<-B and B<-A directions.

        """
        #Rename regrouped files to min.A and min.B to pass as input to PATHSAMPLE
        files_to_modify = [self.path/'min.A', self.path/'min.B',
                           self.path/'min.data', self.path/'ts.data']
        for f in files_to_modify:
            os.system(f"mv {f} {f}.original")
            os.system(f"mv {f}.regrouped.{temp:.10f} {f}")
        #run NGT without regroup on regrouped minima
        self.parse.comment_input('REGROUPFREE')
        self.parse.comment_input('DUMPGROUPS')
        self.parse.append_input('NGT', '0 T')
        self.parse.write_input(scan.pathdatafile)
        outfile = open(self.path/f'out.NGT.kNGT.{Gthresh:.2f}','w')
        subprocess.run(f"{PATHSAMPLE}", stdout=outfile, cwd=self.path)
        self.parse.parse_output(outfile=self.path/f'out.NGT.kNGT.{Gthresh:.2f}')
        rates = {}
        rates['kAB'] = self.parse.output['kAB']
        rates['kBA'] = self.parse.output['kBA']
        rates['MFPTAB'] = self.parse.output['MFPTAB']
        rates['MFPTBA'] = self.parse.output['MFPTBA']
        rates['kNSSAB'] = self.parse.output['kNSSAB']
        rates['kNSSBA'] = self.parse.output['kNSSBA']
        rates['kSSAB'] = self.parse.output['kSSAB']
        rates['kSSBA'] = self.parse.output['kSSBA']
        #restore original file names
        for f in files_to_modify:
            os.system(f"mv {f} {f}.regrouped.{temp:.10f}")
            os.system(f"mv {f}.original {f}")
        return rates

    def run_NGT_exact(self, temp):
        """Run a single GT calculation using the PATHSAMPLE program 
        at the specified temperature `temp` with inputs specified by the self.parse class."""
        self.parse.comment_input('REGROUPFREE')
        self.parse.comment_input('DUMPGROUPS')
        self.parse.append_input('NGT', '0 T')
        self.parse.append_input('TEMPERATURE', temp)
        self.parse.write_input(self.pathdatafile)
        outfile = open(self.path/'out.NGT.NOREGROUP','w')
        subprocess.run(f"{PATHSAMPLE}", stdout=outfile, cwd=self.path)
        self.parse.parse_output(outfile=self.path/'out.NGT.NOREGROUP')
        rates = {}
        rates['kAB'] = self.parse.output['kAB']
        rates['kBA'] = self.parse.output['kBA']
        rates['MFPTAB'] = self.parse.output['MFPTAB']
        rates['MFPTBA'] = self.parse.output['MFPTBA']
        rates['kNSSAB'] = self.parse.output['kNSSAB']
        rates['kNSSBA'] = self.parse.output['kNSSBA']
        rates['kSSAB'] = self.parse.output['kSSAB']
        rates['kSSBA'] = self.parse.output['kSSBA']
        return rates

    def run_regroup(self, temp, value, NGTpostregroup=False):
        """Run REGROUPFREE once at a threshold given by `value`
        and a temperature given by `temp` 
        
        Parameters
        ----------
        temp : float
            Temperature.
        value : float
            Free energy regrouping threshold for REGROUPFREE.
        NGTpostregroup : bool
            If True, a single GT calculation is run on the regrouped network.
            Defaults to False.

        Returns
        -------
        df : pandas DataFrame
            single row with columns temp, Gthresh, and rates computed on regrouped network

        """

        df = pd.DataFrame()
        #update input
        self.parse.append_input('TEMPERATURE', temp)
        self.parse.append_input('REGROUPFREE', value)
        self.parse.append_input('DUMPGROUPS', '')
        self.parse.append_input('PLANCK', 1)
        if NGTpostregroup:
            self.parse.comment_input('NGT')
        else:
            self.parse.append_input('NGT', '0 T')
        #overwrite pathdata file with updated input
        self.parse.write_input(self.pathdatafile)
        #run calculation
        outfile = open(self.path/f'{self.outbase}.REGROUPFREE.{value:.2f}','w')
        subprocess.run(f"{PATHSAMPLE}", stdout=outfile, cwd=self.path)
        #parse output
        self.parse.parse_output(outfile=self.path/f'{self.outbase}.REGROUPFREE.{value:.2f}')
        if NGTpostregroup:
            rates = self.run_NGT_regrouped(value, temp)
            df['kNSSAB'] = rates['kNSSAB']
            df['kNSSBA'] = rates['kNSSBA']
            df['kSSAB'] = rates['kSSAB']
            df['kSSBA'] = rates['kSSBA']
            df['kAB'] = rates['kAB']
            df['kBA'] = rates['kBA']
        else:
            df['kAB_LEA'] = [self.parse.output['kAB']]
            df['kBA_LEA'] = [self.parse.output['kBA']]
        df['Gthresh'] = [value]
        df['T'] = [temp]
        #extract group assignments for this temperature/Gthresh
        communities = self.parse.parse_dumpgroups(self.path/f'minima_groups.{temp:.10f}',
                                    grouptomin=True)
        #create new parse object, parse min.A.regrouped, min.B.regrouped
        ABparse = ParsedPathsample(self.pathdatafile)
        ABparse.parse_minA_and_minB(self.path/f'min.A.regrouped.{temp:.10f}',
                                    self.path/f'min.B.regrouped.{temp:.10f}')
        #calculate the total number of minima in A and B
        regroupedA = []
        for a in ABparse.minA: #0-indexed so add 1
            #count the number of minima in that group, increment total count
            regroupedA += communities[a+1]
        sizeOfA = len(regroupedA)
        regroupedB = []
        for b in ABparse.minB: #0-indexed so add 1
            #count the number of minima in that group, increment total count
            regroupedB += communities[b+1]
        sizeOfB = len(regroupedB)
        df['ncomms'] = len(communities)
        df['regroupedA'] = sizeOfA
        df['regroupedB'] = sizeOfB
        # rename files to also include Gthresh in filename
        files_to_rename = [self.path/f'minima_groups.{temp:.10f}',
                            self.path/f'ts_groups.{temp:.10f}',
                            self.path/f'min.A.regrouped.{temp:.10f}',
                            self.path/f'min.B.regrouped.{temp:.10f}',
                            self.path/f'min.data.regrouped.{temp:.10f}',
                            self.path/f'ts.data.regrouped.{temp:.10f}']
        for f in files_to_rename:
            dir_name = Path(self.path/f'G{value:.1f}')
            if not dir_name.is_dir():
                dir_name.mkdir()
            os.system(f'mv {f} {dir_name}/{f.name}.G{value:.1f}')
        print(f"Computed rate constants for regrouped minima with threshold {value}")
        return df

    def scan_regroup(self, values, temp, NGTpostregroup=False):
        """ Run the REGROUPFREE routine for a range of regrouping thresholds.
        Calculate A<->B rates on regrouped networks and original network and
        save to a csv file.

        Parameters
        ----------
        values : array-like
            Free energy regrouping thresholds.
        temp : float
            Temperature.
        NGTpostregroup : bool
            If True, a single GT calculation is run on each regrouped network.
            Defaults to False.

        """
        
        csv = Path(f'csvs/rates_{self.suffix}.csv')
        dfs = []
        for value in values:
            df = self.run_regroup(temp, value, NGTpostregroup)
            dfs.append(df)
        bigdf = pd.concat(dfs, ignore_index=True, sort=False)
        rates = self.run_NGT_exact()
        bigdf['kNGTexactAB'] = rates['kAB']
        bigdf['kNGTexactBA'] = rates['kBA']
        #if file exists, append to existing data
        if csv.is_file():
            olddf = pd.read_csv(csv)
            bigdf = olddf.append(bigdf)
        #write updated file to csv
        bigdf.to_csv(csv, index=False)

    def scan_temp(self, values, name='TEMPERATURE'):
        """Run PATHSAMPLE over a range of temperatures specified by `values` and
        extract rates from the NGT keyword output which are written to a csv file. 
        This function can also be used to scan any other single keyword by specifying
        the `name` and a range of values.
        """
        corrected_name = str(name).upper()
        csv = Path(f'rates_{self.suffix}.csv')
        dfs = []
        for value in values:
            df = pd.DataFrame()
            #update input
            self.parse.append_input(name, value)
            #overwrite pathdata file with updated input
            self.parse.write_input(self.pathdatafile)
            #run calculation
            outfile = open(self.path/f'{self.outbase}.{name}.{value:.2f}','w')
            subprocess.run(f"{PATHSAMPLE}", stdout=outfile, cwd=self.path)
            #parse output
            self.parse.parse_output(outfile=self.path/f'{self.outbase}.{name}.{value:.2f}')
            #store the output under the value it was run at
            self.outputs[value] = self.parse.output
            df['kNSSAB'] = [self.parse.output['kNSSAB']]
            df['kNSSBA'] = [self.parse.output['kNSSBA']]
            df['kSSAB'] = [self.parse.output['kSSAB']]
            df['kSSBA'] = [self.parse.output['kSSBA']]
            df['kAB'] = [self.parse.output['kAB']]
            df['kBA'] = [self.parse.output['kBA']]
            df['T'] = [value]
            dfs.append(df)
            print(f"Computed rate constants for temperature {value}")
        bigdf = pd.concat(dfs, ignore_index=True, sort=False)
        bigdf['numInA'] = self.parse.numInA
        bigdf['numInB'] = self.parse.numInB
        #if file exists, append to existing data
        if csv.is_file():
            olddf = pd.read_csv(csv)
            bigdf = olddf.append(bigdf)
        #write updated file to csv
        bigdf.to_csv(csv, index=False)

    def remove_output(self):
        """Delete PATHSAMPLE log files."""
        for f in glob.glob(str(self.path/'out*')):
            if Path(f).exists():
                Path(f).unlink()

""" Functions for using the ScanPathsample and ParseSample classes to perform
useful tasks. """

def scan_product_states(numinAs, numinBs, temps):
    """Calculate rates using NGT for various definitions of A and
    B sets."""
    if len(numinAs) != len(numinBs):
        raise ValueError('numinAs and numinBs must have the same shape')

    suffix = 'kNGT_ABscan'
    olddf = pd.read_csv(f'rates_{suffix}.csv')
    olddf = olddf.set_index(['numInA', 'numInB'])
    for i in range(len(numinAs)):
        if (numinAs[i], numinBs[i]) in olddf.index:
            continue
        scan = ScanPathsample('/scratch/dk588/databases/LJ38.2010/10000.minima/pathdata', suffix=suffix)
        #since min.A/min.B have been sorted, simply change scan.parse.numInA/B
        scan.parse.define_A_and_B(numinAs[i], numinBs[i], sorted=True, mindata=scan.path/'min.data')
        #write new min.A/B files with first line changed
        scan.parse.write_minA_minB(scan.path/'min.A',scan.path/'min.B')
        scan.scan_temp('TEMPERATURE', temps)
        scan.remove_output()
        print(f'Num in A: {numinAs[i]}, Num in B: {numinBs[i]}')

def scan_Gthresh_and_temp(temps, nrgthreshs):
    """ Calculate rates at different free energy regrouping thresholds and
    at different temperatures. """

    suffix = 'Gthresh_temp_scan3'
    csv = Path(f'csvs/rates_{suffix}.csv')
    olddf = pd.read_csv(csv)
    olddf = olddf.set_index(['T','Gthresh'])
    for temp in temps:
        dfs = []
        for thresh in nrgthreshs:
            if (temp, thresh) in olddf.index:
                continue
            scan = ScanPathsample('./pathdata', suffix=suffix)
            df = scan.run_regroup(temp, thresh)
            dfs.append(df)
            scan.remove_output()
        bigdf = pd.concat(dfs, ignore_index=True, sort=False)
        scan = ScanPathsample('./pathdata', suffix=suffix)
        rates = scan.run_NGT_exact()
        bigdf['kNGTexactAB'] = rates['kAB']
        bigdf['kNGTexactBA'] = rates['kBA']
        bigdf = bigdf.set_index(['T', 'Gthresh'])
        olddf = pd.concat([olddf, bigdf], axis=0, ignore_index=False)
        scan.remove_output()
    #write updated file to csv
    olddf.to_csv('csvs/rates_Gthresh_temp_scan4.csv')

def dump_ktn_info_scan(path, temps, nrgthreshs=None):
    """Dump `stat_prob.dat`, `ts_conns.dat`, and `ts_weights.dat` files for each
    temperature in `temps`, and one `communites.dat` file for each (temperature, Gthresh) pair.
    
    Parameters
    ----------
    temps : array-like
        Temperatures at which to dump rate matrix and stationary probabilities with DUMPINFO keyword.
    nrgthreshs: array-like
        Free energy regrouping thresholds used for previous runs of REGROUPFREE. Defaults to None.

    """
    scan = ScanPathsample(Path(path)/'pathdata')
    for temp in temps:
        scan.dump_rates_full_network(temp)
        if nrgthreshs is None:
            continue
        else:
            for thresh in nrgthreshs:
                communities = scan.parse.parse_dumpgroups(scan.path/f'G{thresh:.1f}/minima_groups.{temp:.10f}.G{thresh:.1f}')
                write_communities(communities, scan.path/f'communities_G{thresh:.2f}_T{temp:.2f}.dat')



