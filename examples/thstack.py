import ROOT


def main():

    ROOT.gROOT.SetBatch()
    hd1 = ROOT.TH1F("d1", "One", 100, -5.0, 5.0)
    h1 = ROOT.TH1F("1", "One", 100, -5.0, 5.0)
    h2 = ROOT.TH1F("2", "One", 100, -5.0, 5.0)
    h3 = ROOT.TH1F("3", "One", 100, -5.0, 5.0)
    r = ROOT.TRandom()
    for i in range(1000):
        h1.Fill(r.Gaus(0, 2.0))
        h2.Fill(r.Gaus(1, 2.0))
        h3.Fill(r.Gaus(-1, 2.0))
        hd1.Fill(r.Gaus(0, 2.0))
        hd1.Fill(r.Gaus(1, 2.0))
        hd1.Fill(r.Gaus(-1, 2.0))
    h1.SetFillColor(ROOT.kRed)
    h2.SetFillColor(ROOT.kBlue)
    h3.SetFillColor(ROOT.kGreen)
    hd1.SetMarkerStyle(ROOT.kFullCircle)
    c = ROOT.TCanvas("c", "c")
    hds = ROOT.THStack("hds", "hs")
    hds.Add(hd1)
    hs = ROOT.THStack("hs", "hs")
    hs.Add(h1)
    hs.Add(h2)
    hs.Add(h3)
    hs.Draw("")
    hds.Draw("PE1,SAME")
    c.SaveAs("thstack.png")
