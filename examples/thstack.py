#!/usr/bin/env python

from mrtools import utilities

import ROOT


def main(ratio_plot: bool = True, ratio_plot_height=200):

    ROOT.gROOT.SetBatch()

    style = utilities.tdr_style()

    hd1 = ROOT.TH1F("d1", "One", 100, -5.0, 5.0)
    h1 = ROOT.TH1F("1", "One", 100, -5.0, 5.0)
    h2 = ROOT.TH1F("2", "One", 100, -5.0, 5.0)
    h3 = ROOT.TH1F("3", "One", 100, -5.0, 5.0)
    r = ROOT.TRandom()
    for _i in range(10000):
        h1.Fill(r.Gaus(0, 2.0))
        h2.Fill(r.Gaus(1, 2.0))
        h3.Fill(r.Gaus(-1, 2.0))
        hd1.Fill(r.Gaus(0, 2.0))
        hd1.Fill(r.Gaus(1, 2.0))
        hd1.Fill(r.Gaus(-1, 2.0))
        hd1.Fill(r.Gaus(0, 2.0))
        hd1.Fill(r.Gaus(1, 2.0))
        hd1.Fill(r.Gaus(-1, 2.0))

    int_dat = hd1.Integral()
    int_bck = h1.Integral() + h2.Integral() + h3.Integral()
    hd1.SetMarkerSize(0.75)
    h1.SetFillColor(ROOT.kRed)
    h2.SetFillColor(ROOT.kBlue)
    h3.SetFillColor(ROOT.kGreen)
    h1.Scale(int_dat / int_bck)
    h2.Scale(int_dat / int_bck)
    h3.Scale(int_dat / int_bck)
    hd1.SetMarkerStyle(ROOT.kFullCircle)
    # c = ROOT.TCanvas("c", "c", 0, 0, 600, 600)
    c = ROOT.TCanvas("c", "c", 0, 0, 600, 800)

    if ratio_plot:
        y_border = ratio_plot_height / c.GetWindowHeight()
        c.Divide(1, 2, 0, 0)

        p1 = c.cd(1)
        p1.SetBottomMargin(0)
        p1.SetLeftMargin(0.15)
        p1.SetTopMargin(0.07)
        p1.SetRightMargin(0.05)
        p1.SetPad(p1.GetX1(), y_border, p1.GetX2(), p1.GetY2())

        p2 = c.cd(2)
        p2.SetTopMargin(0)
        p2.SetRightMargin(0.05)
        p2.SetLeftMargin(0.15)
        p2.SetBottomMargin(0.25)  # 0.07
        p2.SetPad(p2.GetX1(), p2.GetY1(), p1.GetX2(), y_border)
    else:
        p1 = c

    p1.cd()

    # for h in (hd1, h1, h2, h3):
    #     h.SetMinimum(0)
    #     h.SetMaximum(300)

    #     h.GetYaxis().SetLimits(0, 200)
    #     h.GetXaxis().SetTitle("X Axis")
    #     h.GetYaxis().SetTitle("Y Axis")
    #     # precision 3 fonts. see https://root.cern.ch/root/htmldoc//TAttText.html#T5
    #     h.GetXaxis().SetTitleFont(43)
    #     h.GetYaxis().SetTitleFont(43)
    #     h.GetXaxis().SetLabelFont(43)
    #     h.GetYaxis().SetLabelFont(43)
    #     h.GetXaxis().SetTitleSize(24)
    #     h.GetYaxis().SetTitleSize(24)
    #     h.GetXaxis().SetLabelSize(20)
    #     h.GetYaxis().SetLabelSize(20)
    #     h.GetXaxis().SetTitleOffset(3.2)
    #     if ratio_plot:
    #         h.GetYaxis().SetTitleOffset(1.6)
    #     else:
    #         h.GetYaxis().SetTitleOffset(1.3)

    hds = ROOT.THStack("hds", "hs")
    hds.SetMinimum(0)
    hds.SetMaximum(2000)
    hds.Add(hd1)
    hs = ROOT.THStack("hs", "hs")
    hs.SetMinimum(0)
    hs.SetMaximum(2000)
    hs.Add(h1)
    hs.Add(h2)
    hs.Add(h3)
    hs.Draw("HIST")
    hds.Draw("PE0, SAME")
    for h in (hs, hds):
        h.GetXaxis().SetTitle("X Axis")
        h.GetYaxis().SetTitle("Super Axis")
        # precision 3 fonts. see https://root.cern.ch/root/htmldoc//TAttText.html#T5
        h.GetXaxis().SetTitleFont(43)
        h.GetYaxis().SetTitleFont(43)
        h.GetXaxis().SetLabelFont(43)
        h.GetYaxis().SetLabelFont(43)
        h.GetXaxis().SetTitleSize(24)
        h.GetYaxis().SetTitleSize(24)
        h.GetXaxis().SetLabelSize(20)
        h.GetYaxis().SetLabelSize(20)
        h.GetYaxis().SetTitleOffset(2.0)

    hs.Draw("HIST")
    hds.Draw("PE1,SAME")

    if ratio_plot:

        hr = hd1 / (h1 + h2 + h3)

        hr.GetYaxis().SetRangeUser(0.1, 1.9)
        hr.SetStats(False)

        hr.GetXaxis().SetTitle("X Axis")
        hr.GetYaxis().SetTitle("Data / MC")

        hr.GetXaxis().SetTitleFont(43)
        hr.GetYaxis().SetTitleFont(43)
        hr.GetXaxis().SetLabelFont(43)
        hr.GetYaxis().SetLabelFont(43)
        hr.GetXaxis().SetTitleSize(24)
        hr.GetYaxis().SetTitleSize(24)
        hr.GetXaxis().SetLabelSize(20)
        hr.GetYaxis().SetLabelSize(20)

        hr.GetXaxis().SetTitleOffset(3.5)
        hr.GetYaxis().SetTitleOffset(2.0)

        hr.GetXaxis().SetTickLength(0.03 * 3)
        hr.GetYaxis().SetTickLength(0.03 * 2)

        hr.GetYaxis().SetNdivisions(505)

        line = ROOT.TLine(hr.GetXaxis().GetXmin(), 1.0, hr.GetXaxis().GetXmax(), 1.0)

        p2.cd()
        hr.Draw()
        line.Draw()

    c.SaveAs("thstack.png")


if __name__ == "__main__":
    main(True)
