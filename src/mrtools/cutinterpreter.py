#!/usr/bin/env python
#
# CutIntepreter for ROOT DataFlow
#
# The cutinterpeter allows to specify a number of event selection criteria as a string
# and transforms the individual tokens into logical expressions for evaluation with
# DataFlow.Filter. The implementation follows a previous implementation for ROOT 
# TTrees.
#
# The selection criteria are specified as tokens seperated by -.  
#
# Each criteria has a name and an associated type (Integer, Float or RegExp). It can be 
# followed by numbers to define limits
#
# Integer criteria 
#     1) nISRJets  declared as I
# 
#     nISRJets1  -> nISRJets == 1
#     nISRJets1p -> nISRJets >= 1
#  
#     2) ntau declared as I:nGoodTau
#
#     ntau0  -> nGoodTau == 0
#
# Float criteria
#
#     1) met -> F:met_pt
#
#     met200 -> met_pt >= 200
#     met200To250 -> met_pt >= 200 && met_pt < 250
# 
# Regexp criteria
# 
#     lepSel -> 
# 
#     lepsel(\d+)
# 
#  
# Example:
#
# nISRJets1p-ntau0-lepSel-dphimetjet0to0p5-jet3Veto-met200-ht300
# nISRJets1p-ntau0-lepSel-deltaPhiMetJetsInv-jet3Veto-met200-ht300
# nISRJets1p-ntau0-lepSel-dphimetjet0p5-jet3Veto-met200-ht300
# nISRJets1p-ntau0-lepSel-deltaPhiJets-jet3Veto-met200-ht300
# nISRJets1p-ntau0-lepSel-jet3Veto-met200-ht300

from typing import Dict, List, Tuple, Iterable
import re

class CutInterpreter:

    regexp: List[Tuple[re.Match, str]]

    def __init__(self, defs: Iterable[Tuple[str, str]]) -> None:

        self.pattern = []
        for pattern, repl in defs.items():
            repl_up = repl.upper()
            if repl_up in ['I', 'I:']: 
                self.regexp.append(
                    (re.compile(f"{pattern}(\d+)")), f"{pattern} == \1"))
                )
                self.pattern.append( 
                    (re.compile(f"{name}(\d+)p"), f"{name} >= \1") 
                )
            elif repl_up.startswith('I:'):
                self.pattern.append( 
                    (re.compile(f"{name}(\d+)"), f"{repl[2:]} == \1") 
                )
                self.pattern.append( 
                    (re.compile(f"{name}(\d+)p"), f"{repl[2:]} >= \1") 
                )
            elif replacement:
                self.regexps.append( 
                    (re.compile(f"{name}([\d\.]+)To([\d\.]+)"), f"{name} >= \1 && {name} < \2") 
                )
                self.regexps.append( 
                    (re.compile(f"{name}([\d\.+])"), f"{name} >= \1") 
                )
            elif cut.startswith('F:'):
                self.regexps.append( 
                    (re.compile(f"{name}([\d\.]+)To([\d\.]+)"), f"{name} >= \1 && {cut[2:]} < \2") 
                )
                self.regexps.append( 
                    (re.compile(f"{name}([\d\.+])"), f"{cut[2:]} >= \1") 
                )
            else:
                self.regexps.append( (re.compile(name), cut) )


if __name__ == '__main__':

    defs = [
        ('nISRJets', 'I'),
        ('ntau', 'I:nGoodTaus'),
        ('lepSel', 'Sum(lep_pt>20)<=1 && l1_pt>0'),
        ('jet3Veto', "nJetGood<=2 || JetGood_pt[2] < 60")
        ('met', 'F:met_pt')
        ('ht', 'F:HT')
    ]
    ci = CutInterpreter(defs)

    ci.print('nISRJets1p-ntau0-lepSel-jet3Veto-met200-ht300')