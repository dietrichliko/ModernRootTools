#!/usr/bin/env python
"""Write yaml files for sample definition."""

import click

PATH = "/store/user/liko/StopsCompressed/nanoTuples"

PERIODES = {
    "Run2016preVFP": "compstops_UL16APVv9_nano_",
    "Run2016postVFP": "compstops_UL16v9_nano_",
    "Run2017": "compstops_UL17v9_nano_",
    "Run2018": "compstops_UL18v9_nano_",
}

LUMINOSITY = {
    "Run2016preVFP": 19.5,
    "Run2016postVFP": 16.5,
    "Run2017": 41.48,
    "Run2018": 59.83,
}
MUON_TRIGGER = {
    "Run2016preVFP": ["HLT_Mu50", "HLT_TkMu50"],
    "Run2016postVFP": ["HLT_Mu50", "HLT_TkMu50"],
    "Run2017": ["HLT_Mu50", "HLT_OldMu100", "HLT_TkMu100"],
    "Run2018": ["HLT_Mu50", "HLT_OldMu100", "HLT_TkMu100"],
}

ELECTRON_TRIGGER = {
    "Run2016preVFP": ["HLT_Ele27_WPTight_Gsf"],
    "Run2016postVFP": ["HLT_Ele27_WPTight_Gsf"],
    "Run2017": ["HLT_Ele35_WPTight_Gsf"],
    "Run2018": ["HLT_Ele32_WPTight_Gsf"],
}

DATA_DATASETS = {
    "Run2016preVFP": [
        "Run2016B_ver2_HIPM_UL",
        "Run2016C_HIPM_UL",
        "Run2016D_HIPM_UL",
        "Run2016E_HIPM_UL",
        "Run2016F_HIPM_UL",
    ],
    "Run2016postVFP": [
        "Run2016F_UL",
        "Run2016G_UL",
        "Run2016H_UL",
    ],
    "Run2017": [
        "Run2017B_UL",
        "Run2017C_UL",
        "Run2017D_UL",
        "Run2017E_UL",
        "Run2017F_UL",
    ],
    "Run2018": [
        "Run2018A_UL",
        "Run2018B_UL",
        "Run2018C_UL",
        "Run2018D_UL",
    ],
}

WJETS_DATASETS = [
    "WJetsToLNu_HT100to200",
    "WJetsToLNu_HT200to400",
    "WJetsToLNu_HT400to600",
    "WJetsToLNu_HT600to800",
    "WJetsToLNu_HT800to1200",
    "WJetsToLNu_HT1200to2500",
    "WJetsToLNu_HT2500toInf",
]

TT_DATASETS = [
    "TTLep_pow_CP5",
    "TTSingleLep_pow_CP5",
]

T_DATASETS = [
    "T_tch_pow",
    "T_tWch_ext",
    "TBar_tch_pow",
    "TBar_tWch_ext",
]

TTX_DATASETS16 = [
    "TTZ_LO",
    "TTGJets",
    "TTWToLNu_CP5",
    "TTWToQQ",
]

TTX_DATASETS = [
    "TTZ_LO",
    "TTGJets",
    "TTW_LO",
]

DY_DATASETS = [
    "DYJetsToLL_M50_HT70to100",
    "DYJetsToLL_M50_HT100to200",
    "DYJetsToLL_M50_HT200to400",
    "DYJetsToLL_M50_HT400to600",
    "DYJetsToLL_M50_HT600to800",
    "DYJetsToLL_M50_HT800to1200",
    "DYJetsToLL_M50_HT1200to2500",
    "DYJetsToLL_M50_HT2500toInf",
]

DYINV_DATASETS = [
    "DYJetsToNuNu_HT100to200",
    "DYJetsToNuNu_HT200to400",
    "DYJetsToNuNu_HT400to600",
    "DYJetsToNuNu_HT600to800",
    "DYJetsToNuNu_HT800to1200",
    "DYJetsToNuNu_HT1200to2500",
    "DYJetsToNuNu_HT2500toInf",
]

QCD_DATASETS = [
    "QCD_HT50to100",
    "QCD_HT100to200",
    "QCD_HT200to300",
    "QCD_HT300to500",
    "QCD_HT500to700",
    "QCD_HT700to1000",
    "QCD_HT1000to1500",
    "QCD_HT1500to2000",
    "QCD_HT2000toInf",
]

SIGNAL_DATASETS = [
    "SMS_T2tt_mStop_500_mLSP_420",
    "SMS_T2tt_mStop_500_mLSP_450",
    "SMS_T2tt_mStop_500_mLSP_470",
]

SIGNAL_TITLE = {
    "SMS_T2tt_mStop_500_mLSP_420": "m_{#tilde{t}}=500, m_{#tilde{#chi}} = 420",
    "SMS_T2tt_mStop_500_mLSP_450": "m_{#tilde{t}}=500, m_{#tilde{#chi}} = 450",
    "SMS_T2tt_mStop_500_mLSP_470": "m_{#tilde{t}}=500, m_{#tilde{#chi}} = 470",
}

COLOR = {
    "WJets": 8,
    "TT": "kAzure+1",
    "T": 7,
    "TTX": "kAzure-7",
    "DY": "kMagenta-6",
    "DYINV": "kOrange-3",
    "QCD": "kMagenta+3",
}


@click.command()
@click.option("--skim", type=click.Choice(["Met", "MetLepEnergy"]), default="Met")
@click.option("--version", default="v5")
def main(skim: str, version: str):
    """Write nanoNtuple yaml files."""
    with open(f"examples/samples/{skim}_nanoNtuple_{version}.yaml", "w") as out:

        for period, period_dir in PERIODES.items():

            print("---", file=out)
            print(f"name: {skim}_nanoNtuple_{version}", file=out)
            print(f"period: {period}", file=out)
            if skim == "Met":
                print("attributes:", file=out)
                print(f"  integrated_luminosity: {LUMINOSITY[period]}", file=out)
            print("samples:", file=out)
            if skim == "Met":
                print("  - name: MET", file=out)
                print("    type: Data", file=out)
                print("    samples:", file=out)
                for dataset in DATA_DATASETS[period]:
                    print(f"      - name: {dataset}", file=out)
                    print(
                        f"        directory: {PATH}/{period_dir}{version}/Met/MET_{dataset}",
                        file=out,
                    )
            elif skim == "MetLepEnergy":
                print("  - name: SingleMuon", file=out)
                print('    title: "Single #mu"', file=out)
                print("    type: Data", file=out)
                print("    attributes:", file=out)
                print(f"      integrated_luminosity: {LUMINOSITY[period]}", file=out)
                print("      trigger:", file=out)
                for trigger in MUON_TRIGGER[period]:
                    print(f"        - {trigger}", file=out)
                print("    samples:", file=out)
                for dataset in DATA_DATASETS[period]:
                    print(f"      - name: {dataset}", file=out)
                    print(
                        f"        directory: {PATH}/{period_dir}{version}/MetLepEnergy/SingleMuon_{dataset}",
                        file=out,
                    )
                print("  - name: SingleElectron", file=out)
                print('    title: "Single e"', file=out)
                print("    type: Data", file=out)
                print("    attributes:", file=out)
                print(f"      integrated_luminosity: {LUMINOSITY[period]}", file=out)
                print("      trigger:", file=out)
                for trigger in ELECTRON_TRIGGER[period]:
                    print(f"        - {trigger}", file=out)
                print("    samples:", file=out)
                for dataset in DATA_DATASETS[period]:
                    print(f"      - name: {dataset}", file=out)
                    print(
                        f"        directory: {PATH}/{period_dir}{version}/MetLepEnergy/SingleElectron_{dataset}",
                        file=out,
                    )

            print("  - name: WJets", file=out)
            print('    title: "W + Jets"', file=out)
            print("    type: Background", file=out)
            print("    attributes:", file=out)
            print(f"      color: {COLOR['WJets']}", file=out)
            print("    samples:", file=out)
            for dataset in WJETS_DATASETS:
                print(f"      - name: {dataset}", file=out)
                print(
                    f"        directory: {PATH}/{period_dir}{version}/{skim}/{dataset}",
                    file=out,
                )

            print("  - name: TT", file=out)
            print('    title: "t#bar{t}"', file=out)
            print("    type: Background", file=out)
            print("    attributes:", file=out)
            print(f"      color: {COLOR['TT']}", file=out)
            print("    samples:", file=out)
            for dataset in TT_DATASETS:
                print(f"      - name: {dataset}", file=out)
                print(
                    f"        directory: {PATH}/{period_dir}{version}/{skim}/{dataset}",
                    file=out,
                )

            print("  - name: T", file=out)
            print('    title: "Single t"', file=out)
            print("    type: Background", file=out)
            print("    attributes:", file=out)
            print(f"      color: {COLOR['TT']}", file=out)
            print("    samples:", file=out)
            for dataset in T_DATASETS:
                print(f"      - name: {dataset}", file=out)
                print(
                    f"        directory: {PATH}/{period_dir}{version}/{skim}/{dataset}",
                    file=out,
                )

            print("  - name: TTX", file=out)
            print('    title: "t#bar{t} #rightarrow X"', file=out)
            print("    type: Background", file=out)
            print("    attributes:", file=out)
            print(f"      color: {COLOR['TTX']}", file=out)
            print("    samples:", file=out)
            if period.startswith("Run2016"):
                for dataset in TTX_DATASETS16:
                    print(f"      - name: {dataset}", file=out)
                    print(
                        f"        directory: {PATH}/{period_dir}{version}/{skim}/{dataset}",
                        file=out,
                    )
            else:
                for dataset in TTX_DATASETS:
                    print(f"      - name: {dataset}", file=out)
                    print(
                        f"        directory: {PATH}/{period_dir}{version}/{skim}/{dataset}",
                        file=out,
                    )

            print("  - name: DY", file=out)
            print('    title: "#gamma/Z^{*}"', file=out)
            print("    type: Background", file=out)
            print("    attributes:", file=out)
            print(f"      color: {COLOR['DY']}", file=out)
            print("    samples:", file=out)
            for dataset in DY_DATASETS:
                print(f"      - name: {dataset}", file=out)
                print(
                    f"        directory: {PATH}/{period_dir}{version}/{skim}/{dataset}",
                    file=out,
                )

            print("  - name: DYINV", file=out)
            print('    title: "#gamma/Z^{*} #rightarrow #nu#bar{#nu}"', file=out)
            print("    type: Background", file=out)
            print("    attributes:", file=out)
            print(f"      color: {COLOR['DYINV']}", file=out)
            print("    samples:", file=out)
            for dataset in DYINV_DATASETS:
                print(f"      - name: {dataset}", file=out)
                print(
                    f"        directory: {PATH}/{period_dir}{version}/{skim}/{dataset}",
                    file=out,
                )

            print("  - name: QCD", file=out)
            print('    title: "QCD"', file=out)
            print("    type: Background", file=out)
            print("    attributes:", file=out)
            print(f"      color: {COLOR['QCD']}", file=out)
            print("    samples:", file=out)
            if period in ["Run2016preVFP", "Run2016postVFP"]:
                for dataset in QCD_DATASETS:
                    print(f"      - name: {dataset}", file=out)
                    print("        samples:", file=out)
                    print(f"          - name: {dataset}", file=out)
                    print(
                        f"            directory: {PATH}/{period_dir}{version}/{skim}/{dataset}",
                        file=out,
                    )
                    print(f"          - name: {dataset}_madgraph", file=out)
                    print(
                        f"            directory: {PATH}/{period_dir}{version}/{skim}/{dataset}_madgraph",
                        file=out,
                    )
            else:
                for dataset in QCD_DATASETS:
                    print(f"      - name: {dataset}", file=out)
                    print(
                        f"        directory: {PATH}/{period_dir}{version}/{skim}/{dataset}",
                        file=out,
                    )

            for dataset in SIGNAL_DATASETS:
                print(f"  - name: {dataset}", file=out)
                print(f'    title: "{SIGNAL_TITLE[dataset]}"', file=out)
                print("    type: Signal", file=out)
                print(
                    f"    directory: {PATH}/{period_dir}{version}/{skim}/{dataset}",
                    file=out,
                )


if __name__ == "__main__":

    main()
