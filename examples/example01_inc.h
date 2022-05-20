#ifndef MRTOOLS_EXAMPLE01_INC_H_
#define MRTOOLS_EXAMPLE01_INC_H_
// Transverse momentum of lepton and MET 
// Usage: pt = pt_lep_met(lep_pt, lep_phi, met_pt, met_phi)
// Lepton can be vector
template<typename T>
T pt_lep_met(const T & pt1, const T & phi1, const float pt2, const float phi2)
{
    return sqrt(pt1 * pt1 + pt2 * pt2 + 2 * pt1 * pt2 * cos(phi1 - phi2));
}

// Transverse mass of lepton and MET 
// Usage: pt = mt_lep_met(lep_pt, lep_phi, met_pt, met_phi)
// Lepton can be vector
template<class T>
T mt_lep_met(const T & pt1, const T & phi1, const float pt2, const float phi2)
{
    return sqrt(2. * pt1 * pt2 * (1. - cos(phi1 - phi2)));
}

#endif