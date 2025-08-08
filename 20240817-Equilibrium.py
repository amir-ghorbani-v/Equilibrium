# <editor-fold desc="######################################## Log">
import sys
Date = "20240817"
# sys.stdout = open(Date + '-Reparameterization.txt', 'w')
# </editor-fold>

# <editor-fold desc="######################################## OPTIONS">
print("######################################## OPTIONS")

PotentialName = "BMD192"

# <editor-fold desc="**********  Library">
print("**********  Library")
import concurrent.futures.process
# import cv2
import datetime
import fileinput
import getpass
import glob
import math
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.patches as patches
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import ntpath
import os  # I/O operations
import pathlib
import platform
import pickle
import shutil
import sys
import time
from pathlib import Path
import pandas as pd  # data process; python-version SQL
import scipy.constants as const
import seaborn as sns
from ase.build import bulk
from ase.calculators.eam import EAM
from scipy import odr
from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator as PchipFunc
from scipy.interpolate import pchip_interpolate as PchipValue
from scipy.interpolate import splrep, splev
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
#from scipy.misc import derivative
# from optimparallel import minimize_parallel
from scipy.optimize import minimize

# </editor-fold>

# <editor-fold desc="**********  Environment">
print("**********  Environment")
print(os.path.dirname(sys.executable))
print(getpass.getuser())
print(platform.system())

# SimulationEnvironement = "MyPc"
# SimulationEnvironement = "ComputeCanada"

if platform.system() == "Windows":
    SimulationEnvironement = "MyPc"

    CurrentDirectory = os.getcwd()
    OriginalPotentialAddress = "D:/Queens_University/Project/Zr/PotentialBank/Eam/BMD192/" + PotentialName + ".eampot"
    LammpsPythonAddress = "C:/Users/19ag40/AppData/Local/LAMMPS 64-bit 2Aug2023 with Python/Python/"
    LammpsTempAddress = "D:/Queens_University/Project/Zr/PotentialDevelopement/LammpsTemp"

    GroupRun = True
    GroupPlot = False
    PlottingSingle = True
    RunLammpsEph = False
    MeasureRhoEos = True
    MeasureRhoQsd = False
    MeasureFDerEos = True
    MeasureFDerQsd = False
    # LammpsPythonAddress = "D:/Setups/LAMMPS/python"

    # <editor-fold desc="LAMMPS on the fly">
    print("LAMMPS on the fly")

    print(LammpsPythonAddress)
    sys.path.append(LammpsPythonAddress)
    from lammps import lammps

    # </editor-fold>

    # <editor-fold desc="Checking LAMMPS">
    print("Checking LAMMPS")
    LammpsScreen = "none"
    LammpsArgs = ["-screen", LammpsScreen]
    LammpsCheck = lammps(cmdargs=LammpsArgs)
    print("LAMMPS Version: ", LammpsCheck.version())
    LammpsCheck.close()
    # </editor-fold>

elif platform.system() == "Linux":
    SimulationEnvironement = "ComputeCanada"

    CurrentDirectory = os.getcwd()
    OriginalPotentialAddress = "/home/veshand/Zr/PotentialBank/Eam/Mendelev/" + PotentialName + ".eampot"
    LammpsPythonAddress = "/home/veshand/.local/easybuild/software/2020/avx512/MPI/intel2020/openmpi4/lammps-eph/20220623"
    LammpsTempAddress = "/home/veshand/Zr/PotentialDevelopement/LammpsTemp"

    GroupRun = True
    GroupPlot = False
    PlottingSingle = False
    RunLammpsEph = True
    MeasureRhoEos = False
    MeasureRhoQsd = False
    MeasureFDerEos = False
    MeasureFDerQsd = False
# </editor-fold>

# <editor-fold desc="**********  Variables">
print("**********  Variables")
StageName = "Options"
EosTemplate = "20240214-Eos.lammpstemp"
QsdTemplate = "20240216-Qsd"
MinimizeMaxiter = 10000
MinimizeTol = 1e-1

Date = "20240304"
Date = "20240228"
StageNameArray = []

Plotting = False
PlottingShow = True
DeepAnalysis = False
WriteLog = True
WriteDump = True
WriteReport = True
ExtrapolationF = True

#Qsd Extract
OriginalQsdEamExtract = False
OriginalQsdEamEphExtract = False
ArturQsdExtract = False
EgtQsdEamExtract = False
EgtSatQsdEamExtract = False
EgtSatZblQsdEamExtract = False
EgtSatZblPpmQsdEamExtract = False
EgtSatZblPpmEpmQsdEamExtract = False

#Eos Extract
OriginalEosEamExtract = False
OriginalEosEamEphExtract = False
ArturEosExtract = False
EgtEosEamExtract = False
EgtSatEosEamExtract = False
EgtSatZblEosEamExtract = False
EgtSatZblPpmEosEamExtract = False
EgtSatZblPpmEpmEosEamExtract = False

DicOnTheFlyOriginal = GroupRun
DicOnTheFlyEgt = GroupRun
DicOnTheFlyEgtSat = GroupRun
DicOnTheFlyEgtSatZbl = GroupRun
DicOnTheFlyEgtSatZblPpm = GroupRun
DicOnTheFlyEgtSatZblPpmEpm = GroupRun

#Qsd Run Lammps
OriginalQsdEamRunLammps = GroupRun#GroupRun
EgtQsdEamRunLammps = GroupRun
EgtSatQsdEamRunLammps = GroupRun
EgtSatZblQsdEamRunLammps = GroupRun
EgtSatZblPpmQsdEamRunLammps = GroupRun
EgtSatZblPpmEpmQsdEamRunLammps = GroupRun

ArturQsdEamRunLammps = GroupRun

#Eos Run Lammps
OriginalEosEamRunLammps = GroupRun
EgtEosEamRunLammps = GroupRun
EgtSatEosEamRunLammps = GroupRun
EgtSatZblEosEamRunLammps = GroupRun
EgtSatZblPpmEosEamRunLammps = GroupRun
EgtSatZblPpmEpmEosEamRunLammps = GroupRun

ArturEosEamRunLammps = False

CoreNumber = mp.cpu_count()
CoreNumber = 2
print("Number of Available cores is:" + str(CoreNumber))

# os.system("pause")

RunLammpsQsdAlongAStart = 0
RunLammpsQsdAlongAFinish = 28

RunLammpsQsdAlongAcStart = 0
RunLammpsQsdAlongAcFinish = 47

RunLammpsQsdAlongCStart = 0
RunLammpsQsdAlongCFinish = 49

RunLammpsEosStart = -23
RunLammpsEosFinish = 60

Factor = 1  # enlarging factor for periodic boundary condition
CutOff = 7.6

XPpm = 0.01
Xmax = 7
# to reduce the computation time, rho measurement is limited between these two numbers:
# for test: 99-100
# For SIA: 1381-1382
CalculatorMin = 1# for test 99
CalculatorMax = 3# for test 101

LatticeEquilibriumDist = 3.234055

ExtractionStartingRow = 0 #132
ExtractionFinishingRow = 0 #2630

BondLengthAlongA = LatticeEquilibriumDist                                         #for Zr
BondLengthAlongA = 4.8439-1.6241
BondLengthAlongC = BondLengthAlongA * 1.598                                      #distance between atoms along C between two A layers in ABABA structure
BondLengthAlongC = 10.3132-5.16793
BondLengthAlongAc = (0**2 + 1.86714**2 + 2.58377**2)**0.5                        #Vector toward 24th atom 3.187801624
# print(BondLengthAlongAc)
EosRev = 0.1
EdRevAlongA = 0.1
EdRevAlongC = 0.1
EdRevAlongAc = 0.02 * BondLengthAlongAc

FolderAddressEosEamOriginal = CurrentDirectory + "/Eos/Original"
FolderAddressEosEamOriginalEph = CurrentDirectory + "/Eos/Original-Eph"
FolderAddressEosEamOriginalEphPc = CurrentDirectory + "/Eos/Original-Pc"
FolderAddressEosEamArtur = CurrentDirectory + "/Eos/Artur"
FolderAddressEosEamEgt = CurrentDirectory + "/Eos/Egt"
FolderAddressEosEamEgtSat = CurrentDirectory + "/Eos/EgtSat"
FolderAddressEosEamEgtSatEph = CurrentDirectory + "/Eos/EgtSat-Eph"
FolderAddressEosEamPpm = CurrentDirectory + "/Eos/EgtSatPpm"
FolderAddressEosEamPpmEph = CurrentDirectory + "/Eos/EgtSatPpm-Eph"
FolderAddressEosEamPpmEpm = CurrentDirectory + "/Eos/EgtSatZblPpmEpm"
FolderAddressEosEamPpmEpmEph = CurrentDirectory + "/Eos/EgtSatZblPpmEpm-Eph"
FolderAddressDftEos = "D:\Queens_University\Project\Zr\PotentialDevelopment\DFT/Eos/CalculationRelax-Kpoints6.4.4"

FolderAddressEamAlongA = CurrentDirectory + "/Qsd/Original/AlongA"
FolderAddressEamAlongAc = CurrentDirectory + "/Qsd/Original/AlongAc"
FolderAddressEamAlongC = CurrentDirectory + "/Qsd/Original/AlongC"

FolderAddressEamEdOriginalPcAlongA = CurrentDirectory + "/Qsd/Original-OnFly/A"
FolderAddressEamEdOriginalPcAlongAc = CurrentDirectory + "/Qsd/Original-OnFly/Ac"
FolderAddressEamEdOriginalPcAlongC = CurrentDirectory + "/Qsd/Original-OnFly/C"

FolderAddressEamEdArturAlongA = CurrentDirectory + "/Qsd/Artur/A"
FolderAddressEamEdArturAlongAc = CurrentDirectory + "/Qsd/Artur/Ac"
FolderAddressEamEdArturAlongC = CurrentDirectory + "/Qsd/Artur/C"

FolderAddressEamEdArturPcAlongA = CurrentDirectory + "/Qsd/Artur-OnFly/A"
FolderAddressEamEdArturPcAlongAc = CurrentDirectory + "/Qsd/Artur-OnFly/Ac"
FolderAddressEamEdArturPcAlongC = CurrentDirectory + "/Qsd/Artur-OnFly/C"

FolderAddressEamEdEgtSatZblAlongA = CurrentDirectory + "/Qsd/EgtSatZbl/A"
FolderAddressEamEdEgtSatZblAlongAc = CurrentDirectory + "/Qsd/EgtSatZbl/Ac"
FolderAddressEamEdEgtSatZblAlongC = CurrentDirectory + "/Qsd/EgtSatZbl/C"

FolderAddressEamEdEgtSatZblPcAlongA = CurrentDirectory + "/Qsd/EgtSatZbl/OnTheFly/A"
FolderAddressEamEdEgtSatZblPcAlongAc = CurrentDirectory + "/Qsd/EgtSatZbl/OnTheFly/Ac"
FolderAddressEamEdEgtSatZblPcAlongC = CurrentDirectory + "/Qsd/EgtSatZbl/OnTheFly/C"

FolderAddressEamEdEgtSatZblPpmAlongA = CurrentDirectory + "/Qsd/EgtSatZblPpm/A"
FolderAddressEamEdEgtSatZblPpmAlongAc = CurrentDirectory + "/Qsd/EgtSatZblPpm/Ac"
FolderAddressEamEdEgtSatZblPpmAlongC = CurrentDirectory + "/Qsd/EgtSatZblPpm/C"

FolderAddressEamEdEgtSatZblPpmPcAlongA = CurrentDirectory + "/Qsd/EgtSatZblPpm-OnFly/A"
FolderAddressEamEdEgtSatZblPpmPcAlongAc = CurrentDirectory + "/Qsd/EgtSatZblPpm-OnFly/Ac"
FolderAddressEamEdEgtSatZblPpmPcAlongC = CurrentDirectory + "/Qsd/EgtSatZblPpm-OnFly/C"

FolderAddressEamEdEgtSatZblPpmEpmAlongA = CurrentDirectory + "/Qsd/EgtSatZblPpmEpm/A"
FolderAddressEamEdEgtSatZblPpmEpmAlongAc = CurrentDirectory + "/Qsd/EgtSatZblPpmEpm/Ac"
FolderAddressEamEdEgtSatZblPpmEpmAlongC = CurrentDirectory + "/Qsd/EgtSatZblPpmEpm/C"

FolderAddressEamEdEgtSatZblPpmEpmPcAlongA = CurrentDirectory + "/Qsd/EgtSatZblPpmEpm-OnFly/A"
FolderAddressEamEdEgtSatZblPpmEpmPcAlongAc = CurrentDirectory + "/Qsd/EgtSatZblPpmEpm-OnFly/Ac"
FolderAddressEamEdEgtSatZblPpmEpmPcAlongC = CurrentDirectory + "/Qsd/EgtSatZblPpmEpm-OnFly/C"

CurrentDirectoryPath = Path(CurrentDirectory)
# print(CurrentDirectoryPath)
# DftDirectory = path.parent.absolute()
DftDirectory = str(CurrentDirectoryPath.parents[1]) + "/Dft"
DftDirectory = "D:/Queens_University/Project/Zr/PotentialDevelopement/Dft"
# print(DftDirectory)

FolderAddressDftAlongA = DftDirectory + "/Original/CalculationSCF-Kpoints6.4.4/AlongA"
FolderAddressDftAlongAc = DftDirectory + "/Original/CalculationSCF-Kpoints6.4.4/AlongAc"
FolderAddressDftAlongC = DftDirectory + "/Original/CalculationSCF-Kpoints6.4.4/AlongC"
FolderAddressDftQsd = DftDirectory + "/Qsd/Scf-644-Nsp-Van-211/20231011-DftExtract.csv"
FolderAddressDftEos = DftDirectory + "/Eos/Relax-777-NSP-50-500/20230418-EosDftExtract.csv"
FolderAddressDftQsdFit = DftDirectory + "/Qsd/Scf-644-Nsp-Van-211/20231011-DftInterpolate.csv"
FolderAddressDftEosFit = DftDirectory + "/Eos/Relax-777-NSP-50-500/20230504-DftSplineTabulated.csv"

Ry2eV = 13.605698066  # unit conversion (eV - Rdy)
IsolatedAtomEv = 98.88545811  # Zr reference energy
RhoAveEqui = 83.27756328   #Calculated from a seperate python file which only goes through the equilibrium

BoxDftToEamX = 1/(6.4677606991648213e+00-2.3930083511736200e-04)
BoxDftToEamY = 1.732400/(1.1204748635232971e+01-4.1456476687518062e-04)
BoxDftToEamZ = 1.598000/(1.0335481597265154e+01-3.8240273446454864e-04)

EpsilonNoughtInFaradMeter = const.epsilon_0   # 8.854187817620389e-12 F⋅m−1 # Vacuum permittivity, commonly denoted ε0 (pronounced as "epsilon nought" or "epsilon zero") is the value of the absolute dielectric permittivity of classical vacuum. Alternatively it may be referred to as the permittivity of free space, the electric constant, or the distributed capacitance of the vacuum
EpsilonNoughtInFaradAngstrom = const.epsilon_0 * 1E-10  # 8.854187817620389e-22 F⋅A−1
EpsilonNought = 55.26349406 * 1e-9 * 1e5  # 55.26349406    e2⋅GeV−1⋅fm−1
ElectronChargeInCoulumb = const.electron_volt  # 1.602176634e-19 C
ElectronCharge = 1
EffectiveNuclearCharge = 39.159  # The effective nuclear charge is the net charge an electron experiences in an atom with multiple electrons. The effective nuclear charge may be approximated by the equation: Zeff = Z - S , Where Z is the atomic number and S is the number of shielding electrons.
NuclearCharge = 40


R1 = 0
Rc = 0

TransZblEamStart = 0.45#0.1#1
TransZblEamFinish = 0.55#0.15#1.5
PhiMinRegionNum = 12

TransPpmZblEamStart = 1
TransPpmZblEamFinish = 1.5
TransPpmMixDftStart = 1.25
TransPpmMixDftFinish = 2.25
TransPpmMixDftNum = 10

TransEamDftStart = 2.25
TransEamDftFinish = 2.5
TransDftEamStart = 2.6
TransDftEamFinish = 2.85
TransDftEamNum = 10

TransEpmEamDftStart = 1
TransEpmEamDftFinish = 2.5
TransEpmDftEamStart = 4.25
TransEpmDftEamFinish = 5
TransEpmMixDftNum = 10

ExtractingDumpRangeQsdEamAlongAStart = 0
ExtractingDumpRangeQsdEamAlongAFinish = 30
ExtractingDumpRangeQsdEamAlongCStart = 0
ExtractingDumpRangeQsdEamAlongCFinish = 50
ExtractingDumpRangeQsdEamAlongAcStart = 0
ExtractingDumpRangeQsdEamAlongAcFinish = 60

EdEosMovingIndexAlongA = 5-1

EdEosInitialPosXAlongA = 1.6241
EdEosInitialPosYAlongA = 2.81355
EdEosInitialPosZAlongA = 5.16793

EdEosMovingIndexAlongC = 5-1
EdEosInitialPosXAlongC = 1.6241
EdEosInitialPosYAlongC = 2.81355
EdEosInitialPosZAlongC = 5.16793

EdEosMovingIndexAlongAc = 5-1
EdEosInitialPosXAlongAc = 1.6241
EdEosInitialPosYAlongAc = 2.81355
EdEosInitialPosZAlongAc = 5.16793

GraphQsdAlongAXMin = 0
GraphQsdAlongAXMax = 3.2
GraphQsdAlongAYMin = -10
GraphQsdAlongAYMax = 250
GraphQsdAlongCXMin = 0
GraphQsdAlongCXMax = 4
GraphQsdAlongCYMin = -10
GraphQsdAlongCYMax = 25
GraphQsdAlongAcXMin = 1
GraphQsdAlongAcXMax = 3.5
GraphQsdAlongAcYMin = -1
GraphQsdAlongAcYMax = 25
# plt.xlim(1, 5.5)
# plt.ylim(-1, 25)

font = {"family": "serif", #serif" "sans-serif" "cursive" "fantasy" "monospace"
        "weight": "bold",
        "size": 10}
matplotlib.rc("font", **font)
matplotlib.rcParams["lines.linewidth"] = 3
matplotlib.rcParams["figure.figsize"] = (5, 5)
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
plt.rc('font', size=15)          # controls default text size
plt.rc('axes', titlesize=15)     # fontsize of the title
plt.rc('axes', labelsize=20)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)    # fontsize of the x-axis ticks
plt.rc('ytick', labelsize=20)    # fontsize of the y-axis ticks
plt.rc('legend', fontsize=15)    # fontsize of the legend

Colors = ["b", "cyan", "darkred", "red", "green", "lime", "w", "#FF5733", "#33FF57", "#5733FF", "#FF5733", "#33FF57", "#5733FF", "#FF5733", "#33FF57", "#5733FF", "#FF5733", "#33FF57", "#5733FF"]
Colors = ["b", "cyan", "white", "darkred", "red", "white", "green", "lime", "w", "#FF5733", "#33FF57", "#5733FF", "#FF5733", "#33FF57", "#5733FF", "#FF5733", "#33FF57", "#5733FF", "#FF5733", "#33FF57", "#5733FF"]

SleepOpen = 0.0
# </editor-fold>

# <editor-fold desc="********** Dictionary">
print("********** Dictionary")

# <editor-fold desc="^^^^ Plotting">
print("^^^^ Plotting")

DistUnit = r"Distance " + "(" + "Å" + r")"

DicVariable = {
    "Eos": {"Dist": {"Variable": "Dist", "VariableEnergy": "DistEnergy","VariableDist": "PhysicalDist","VariableGridName": "PhysicalDist",
                    "EamAxisScale": "linear", "EamAxisXLabel": DistUnit, "EamAxisYLabel": "Energy","EamAxisXMin": -0.2,"EamAxisXMax": 16,"EamAxisYMin": -20,"EamAxisYMax": 1e3,
                    "FunctionAxisScale": "linear", "FunctionAxisXLabel": DistUnit, "FunctionAxisYLabel": "Energy","FunctionAxisXMin": -1,"FunctionAxisXMax": 8,"FunctionAxisYMin": -20, "FunctionAxisYMax": 140,
                    "SkipHeader": 1,
                    "FunctionName": "F", "FunctionInterpolateName": "DistFInterpolate",  "FunctionInterpolateValuesName": "DistFInterpolateValues"
                     },
            "Rho": {"Variable": "Rho","VariableEnergy": "RhoEnergy","VariableDist": "RhoDist","VariableGridName": "RhoDist",
                    "EamAxisScale": "linear", "EamAxisXLabel": "Rho", "EamAxisYLabel": "Energy","EamAxisXMin": -0.2,"EamAxisXMax": 15,"EamAxisYMin": -5,"EamAxisYMax": 1e2,
                    "FunctionAxisScale": "linear", "FunctionAxisXLabel": DistUnit, "FunctionAxisYLabel": "Energy", "FunctionAxisXMin": -1, "FunctionAxisXMax": 13, "FunctionAxisYMin": -20, "FunctionAxisYMax": 150,
                    "SkipHeader": 2,
                    "FunctionName": "F", "FunctionInterpolateName": "RhoFInterpolate", "FunctionInterpolateValuesName": "RhoFInterpolateValues",
                    }
            },
    "Qsd": {"Dist": {"Variable": "Dist","VariableEnergy": "DistEnergy","VariableDist": "PhysicalDist","VariableGridName": "PhysicalDist",
                    "EamAxisScale": "log", "EamAxisXLabel": DistUnit, "EamAxisYLabel": "Energy","EamAxisXMin": -1,"EamAxisXMax": 5.5,"EamAxisYMin": 1e-4,"EamAxisYMax": 1e2,
                    "FunctionAxisScale": "linear", "FunctionAxisX": DistUnit, "FunctionAxisY": "Energy", "FunctionAxisXMin": -1, "FunctionAxisXMax": 8, "FunctionAxisYMin": 0, "FunctionAxisYMax": 1e3,
                    "SkipHeader": 1,
                    "FunctionName": "Phi", "FunctionInterpolateName": "DistPhiInterpolate",  "FunctionInterpolateValuesName": "DistPhiInterpolateValues"
                     },
            "Rho": {"Variable": "Rho","VariableEnergy": "RhoEnergy","VariableDist": "RhoDist","VariableGridName": "RhoDist",
                    "EamAxisScale": "log", "EamAxisXLabel": "Rho", "EamAxisYLabel": "Energy","EamAxisXMin": -1,"EamAxisXMax": 8,"EamAxisYMin": 1e-4,"EamAxisYMax": 1e2,
                    "FunctionAxisScale": "linear", "FunctionAxisXLabel": DistUnit, "FunctionAxisYLabel": "Energy", "FunctionAxisXMin": -1, "FunctionAxisXMax": 8, "FunctionAxisYMin": 0, "FunctionAxisYMax": 1e3,
                    "SkipHeader": 2,
                    "FunctionName": "Phi", "FunctionInterpolateName": "RhoPhiInterpolate",  "FunctionInterpolateValuesName": "RhoPhiInterpolateValues",
                    }
            },
}
# </editor-fold>

# <editor-fold desc="^^^^ Stages">
print("^^^^ Stages")

DicStage = {
    "Original": {
        "Eos":
            {"Extract": OriginalEosEamExtract, "Run": OriginalEosEamRunLammps, "Read": True},
        "Qsd":
            {"Extract": OriginalQsdEamExtract, "Run": OriginalQsdEamRunLammps, "Read": True}
    },
    "Egt": {
        "Eos":
            {"Extract": EgtEosEamExtract, "Run": EgtEosEamRunLammps, "Read": True},
        "Qsd":
            {"Extract": EgtQsdEamExtract, "Run": EgtQsdEamRunLammps, "Read": True}
    },
    "EgtSat": {
        "Eos":
            {"Extract": EgtSatEosEamExtract, "Run": EgtSatEosEamRunLammps, "Read": True},
        "Qsd":
            {"Extract": EgtSatQsdEamExtract, "Run": EgtSatQsdEamRunLammps, "Read": True}
    },
    "EgtSatZbl": {
        "Eos":
            {"Extract": EgtSatZblEosEamExtract, "Run": EgtSatZblEosEamRunLammps, "Read": True},
        "Qsd":
            {"Extract": EgtSatZblQsdEamExtract, "Run": EgtSatZblQsdEamRunLammps, "Read": True}
    },
    "EgtSatZblPpm": {
        "Eos":
            {"Extract": EgtSatZblPpmEosEamExtract, "Run": EgtSatZblPpmEosEamRunLammps, "Read": True},
        "Qsd":
            {"Extract": EgtSatZblPpmQsdEamExtract, "Run": EgtSatZblPpmQsdEamRunLammps, "Read": True}
    },
    "EgtSatZblPpmEpm": {
        "Eos":
            {"Extract": EgtSatZblPpmEpmEosEamExtract, "Run": EgtSatZblPpmEpmEosEamRunLammps, "Read": True},
        "Qsd":
            {"Extract": EgtSatZblPpmEpmQsdEamExtract, "Run": EgtSatZblPpmEpmQsdEamRunLammps, "Read": True}
    }
}

DicOnTheFly = {
    "Original": DicOnTheFlyOriginal,
    "Egt": DicOnTheFlyEgt,
    "EgtSat": DicOnTheFlyEgtSat,
    "EgtSatZbl": DicOnTheFlyEgtSatZbl,
    "EgtSatZblPpm": DicOnTheFlyEgtSatZblPpm,
    "EgtSatZblPpmEpm": DicOnTheFlyEgtSatZblPpmEpm
}
# </editor-fold>

# <editor-fold desc="^^^^ FittingNumpy">
# <editor-fold desc="DicFittingNumpyStage">
DicFittingNumpy = {
    "": {"Dist": "", "Rho": "", "F": "", "Phi": ""}
}
# </editor-fold>

# <editor-fold desc="DicFittingNumpyInterpolateStage">
DicFittingNumpyInterpolate = {
    "": {"DistRho": "", "DistF": "", "RhoF": "", "DistPhi": ""}
}
# </editor-fold>

# <editor-fold desc="DicFittingNumpyInterpolateValueStage">
DicFittingNumpyInterpolateValue = {
    "": {"Dist": "", "Rho": "", "DistRho": "", "DistF": "", "RhoF": "", "DistPhi": ""}
}
# </editor-fold>
# </editor-fold>

# <editor-fold desc="^^^^ Eam">
print("^^^^ Eam")

# <editor-fold desc="Eam-Points">
print("Eam-Points")
DicEamDf = {
    "": {
        "A": {"Dist": "", "Rho": "", "DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "", "RhoEnergy": "", "FDer": ""},
        "C": {"Dist": "", "Rho": "", "DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "", "RhoEnergy": "", "FDer": ""},
        "Ac": {"Dist": "", "Rho": "", "DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "", "RhoEnergy": "", "FDer": ""},
        "Eos": {"Dist": "", "Rho": "", "DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "", "RhoEnergy": "", "FDer": ""},
    }
}
# </editor-fold>

# <editor-fold desc="Eam-Interpolate">
print("Eam-Interpolate")
#Here, Dist means DistEnergy and Rho means RhoEnergy
DicEamDfInterpolate = {
    "": {
        "A": {"DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "", "RhoEnergy": ""},
        "C": {"DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "", "RhoEnergy": ""},
        "Ac": {"DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "", "RhoEnergy": ""},
        "Eos": {"DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "", "RhoEnergy": ""}
    }
}
# </editor-fold>

# <editor-fold desc="Eam-InterpolateValue">
print("Eam-InterpolateValue")
DicEamDfInterpolateValue = {
    "": {
    "A": {"Dist": "", "Rho": "", "DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "", "RhoEnergy": ""},
     "C": {"Dist": "", "Rho": "", "DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "", "RhoEnergy": ""},
     "Ac": {"Dist": "", "Rho": "", "DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "", "RhoEnergy": ""},
     "Eos": {"Dist": "", "Rho": "", "DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "", "RhoEnergy": ""},
    }
}
# </editor-fold>

# <editor-fold desc="Critical">
print("Critical")
DicEamCritical = {
    "": {
        "A": {"Step": {"Min":"","Max":"","Eq":""}, "Dist": {"Min":"","Max":"","Eq":""}, "Rho":{"Min":"","Max":"","Eq":""}, "Energy":{"Min":"","Max":"","Eq":""}},
        "C": {"Step": {"Min":"","Max":"","Eq":""}, "Dist": {"Min":"","Max":"","Eq":""}, "Rho":{"Min":"","Max":"","Eq":""}, "Energy":{"Min":"","Max":"","Eq":""}},
        "Ac": {"Step": {"Min":"","Max":"","Eq":""}, "Dist": {"Min":"","Max":"","Eq":""}, "Rho":{"Min":"","Max":"","Eq":""}, "Energy":{"Min":"","Max":"","Eq":""}},
        "Eos": {"Step": {"Min":"","Max":"","Eq":""}, "Dist": {"Min":"","Max":"","Eq":""}, "Rho":{"Min":"","Max":"","Eq":""}, "Energy":{"Min":"","Max":"","Eq":""}}
    }
}

# </editor-fold>

# </editor-fold>

# <editor-fold desc="^^^^ Dft">
print("^^^^ Dft")

# <editor-fold desc="Dft-Points">
print("Dft-Points")
DicDftDf = {
    "": {
        "A": {"Dist": "", "Rho": "", "DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "", "RhoEnergy": ""},
        "C": {"Dist": "", "Rho": "", "DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "", "RhoEnergy": ""},
        "Ac": {"Dist": "", "Rho": "", "DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "", "RhoEnergy": ""},
        "Eos": {"Dist": "", "Rho": "", "DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "", "RhoEnergy": ""},
    }
}
# </editor-fold>

# <editor-fold desc="Dft-Interpolate">
print("Dft-Interpolate")
DicDftDfInterpolate = {
    "": {
        "A": {"DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "", "RhoEnergy":""},
        "C": {"DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "", "RhoEnergy":""},
        "Ac": {"DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "", "RhoEnergy":""},
        "Eos": {"DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "", "RhoEnergy":""}
    }
}
# </editor-fold>

# <editor-fold desc="Dft-InterpolateValue">
print("Dft-InterpolateValue")
DicDftDfInterpolateValues = {
    "": {
    "A": {"Dist": "", "Rho": "", "DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "", "RhoEnergy": ""},
     "C": {"Dist": "", "Rho": "", "DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "", "RhoEnergy": ""},
     "Ac": {"Dist": "", "Rho": "", "DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "", "RhoEnergy": ""},
     "Eos": {"Dist": "", "Rho": "", "DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "", "RhoEnergy": ""},
    }
}
# </editor-fold>

# </editor-fold>

# <editor-fold desc="^^^^ Cost">
print("^^^^ Cost")

# <editor-fold desc="RunLammpsRange">
print("RunLammpsRange")

DicRunLammpsRange = {
        "A": {"Step": {"Start": RunLammpsQsdAlongAStart, "Finish": RunLammpsQsdAlongAFinish}, "Dist": {"Start": "", "Finish": ""}, "Rho": {"Start": "", "Finish": ""}},
        "C": {"Step": {"Start": RunLammpsQsdAlongCStart, "Finish": RunLammpsQsdAlongCFinish}, "Dist": {"Start": "", "Finish": ""}, "Rho": {"Start": "", "Finish": ""}},
        "Ac": {"Step": {"Start": RunLammpsQsdAlongAcStart, "Finish": RunLammpsQsdAlongAcFinish}, "Dist": {"Start": "", "Finish": ""}, "Rho": {"Start": "", "Finish": ""}},
        "Eos": {"Step": {"Start": RunLammpsEosStart, "Finish": RunLammpsEosFinish}, "Dist": {"Start": "", "Finish": ""}, "Rho": {"Start": "", "Finish": ""}}
}
# </editor-fold>

# <editor-fold desc="DicCost">
print("DicCost")
DicCostRange = {
        "A": {""},
        "C": {""},
        "Ac": {""},
        "Eos": {""}
}
# </editor-fold>

# <editor-fold desc="Rev">
print("Rev")
DicRev = {
    "A": EdRevAlongA,
    "C": EdRevAlongC,
    "Ac": EdRevAlongAc,
    "Eos": EosRev
}

# </editor-fold>
# </editor-fold>

# <editor-fold desc="^^^^ Potential">
print("^^^^ Potential")

DicPotential = {
    "": {
        "PhysicalDist": "", "RhoDist": "", "Rho": "", "F": "", "Phi": "",
        "DistRhoInterpolate": "", "RhoFInterpolate": "", "DistPhiInterpolate": "",
        "DistRhoInterpolateValues": "", "RhoFInterpolateValues": "", "DistPhiInterpolateValues": ""
    }
}

# </editor-fold>

# <editor-fold desc="^^^^ Input">
print("^^^^ Input")
DicMeasureRho = {
    "Eos": MeasureRhoEos, "Qsd": MeasureRhoQsd
}

DicMeasureFDer = {
    "Eos": MeasureFDerEos, "Qsd": MeasureFDerQsd
}

DicTypeList = {
    "Eos": ["Eos"], "Qsd": ["A", "C", "Ac"]
}
# </editor-fold>

# </editor-fold>

# <editor-fold desc="**********  Functions">
print("********** Functions")

# <editor-fold desc="Equilibrium">
def EquilibriumFunc(Input_V,Pot_V,Directory_V):
    global Num
    LammpsTempFileAddress = "20240817-Equilibrium.lammpstemp"
    LammpsInFileAddress = Directory_V + "/" + "20240817-Equilibrium-" + str(Num) + ".lammpsin"
    LammpsLogFileAddress = Directory_V + "/" + "20240817-Equilibrium-" + str(Num) + ".lammpslog"
    shutil.copyfile(LammpsTempFileAddress, LammpsInFileAddress)

    with fileinput.FileInput(LammpsInFileAddress, inplace=True) as file:
        for line in file:
            line = line.replace("NumTemp", str(Num))
            line = line.replace("PotTemp", str(Pot_V))
            line = line.replace("DirectoryTemp", str(Directory_V))
            line = line.replace("BoxXTemp", str(Input_V[0]))
            line = line.replace("BoxYTemp", str(Input_V[1]))
            line = line.replace("BoxZTemp", str(Input_V[2]))
            print(line, end="")

    Type = "RunBash"
    if Type=="RunWindows":
        LammpsArgs = ["-screen", LammpsScreen]
        lmp = lammps(cmdargs=LammpsArgs)
        lmp.file(LammpsInFileAddress)

    elif Type=="RunBash":
        import subprocess
        srun_command = [
            "mpirun",
            "-np",
            "16",
            "/lustre09/project/6014745/shared/mtp/bin/mtpgpu/lmp_mpi",
            "-in", LammpsInFileAddress,
            "-log", LammpsLogFileAddress
        ]
        #print(srun_command)
        subprocess.run(srun_command, check=True)

    EquilibriumReport = pd.read_csv(Directory_V + "/" + "Equilibrium.csv")

    PressureX = EquilibriumReport["PressureX"].iloc[0]
    PressureY = EquilibriumReport["PressureY"].iloc[0]
    PressureZ = EquilibriumReport["PressureZ"].iloc[0]

    # print(PressureX)
    # print(PressureY)
    # print(PressureZ)
    PressureSum = abs(PressureX) + abs(PressureY) + abs(PressureZ)
    # print(PressureSum)
    Num += 1

    # <editor-fold desc="Save">
    global IterationDf
    print(str(Num), str(Input_V[0]), str(Input_V[1]), str(Input_V[2]))
    IterationNew = pd.DataFrame({'Num': [Num], 'A1X': [Input_V[0]], 'A2Y': [Input_V[1]], 'A3Z': [Input_V[2]],
                                 'PressureX': [PressureX], 'PressureY': [PressureY], 'PressureZ': [PressureZ],
                                 'PressureSum': [PressureSum]})
    IterationNew["Potential"] = Pot
    IterationDf = pd.concat((IterationDf, IterationNew), ignore_index=True)
    # </editor-fold>

    return PressureSum
# </editor-fold>

# <editor-fold desc="add_boundary_knots">
def add_boundary_knots(spline,Order_V,ExtrapolateJump_V):
    """
    Add knots infinitesimally to the left and right.

    Additional intervals are added to have zero 2nd and 3rd derivatives,
    and to maintain the first derivative from whatever boundary condition
    was selected. The spline is modified in place.
    """
    # determine the slope at the left edge
    leftx = spline.x[0]
    lefty = spline(leftx)
    leftslope = spline(leftx, nu=Order_V)

    # add a new breakpoint just to the left and use the
    # known slope to construct the PPoly coefficients.
    leftxnext = np.nextafter(leftx, leftx - ExtrapolateJump_V)
    leftynext = lefty + leftslope*(leftxnext - leftx)
    leftcoeffs = np.array([0, 0, leftslope, leftynext])
    spline.extend(leftcoeffs[..., None], np.r_[leftxnext])

    # repeat with additional knots to the right
    rightx = spline.x[-1]
    righty = spline(rightx)
    # print("Right Point: x=" + str(rightx) + " Y=" + str(righty))
    rightslope = spline(rightx,nu=Order_V)
    rightxnext = rightx + ExtrapolateJump_V
    # print("rightxnext is :" + str(rightxnext))
    rightynext = righty + rightslope * (ExtrapolateJump_V)
    rightcoeffs = np.array([0, 0, rightslope, rightynext])
    spline.extend(rightcoeffs[..., None], np.r_[rightxnext])
# </editor-fold>

# <editor-fold desc="DafaFrame AverageInsertion">
def AverageInsertion(DataFrame_V, Iteration_V):
    DataFrameNew = DataFrame_V.copy()  # Use copy to avoid modifying the original DataFrame
    DataFrameHeader = list(DataFrameNew)
    # print(DataFrameNew)
    for i in range(Iteration_V):
        new_rows = []
        for Row in range(DataFrameNew.shape[0] - 1):
            RowDicNew = {}
            for Column in DataFrameHeader:
                Ave = (DataFrameNew.iloc[Row][Column] + DataFrameNew.iloc[Row + 1][Column]) / 2
                RowDicNew[Column] = Ave
            new_rows.append(RowDicNew)
        DataFrameNew = pd.concat([DataFrameNew, pd.DataFrame(new_rows)], ignore_index=True)
        if 'Step' in DataFrameNew.columns:
            DataFrameNew = DataFrameNew.sort_values(by='Step').reset_index(drop=True)
        else:
            DataFrameNew = DataFrameNew.sort_index().reset_index(drop=True)
    # print(DataFrameNew)
    return DataFrameNew
# </editor-fold>

# <editor-fold desc="PBCDist">
def PBCDist(x1, y1, z1, x2, y2, z2, BoxX, BoxY, BoxZ):
    DX = x2 - x1
    DY = y2 - y1
    DZ = z2 - z1
    RatioX = DX / BoxX
    RatioY = DY / BoxY
    RatioZ = DZ / BoxZ
    if (abs(DX) > BoxX / 2):
        DX = DX - round(RatioX) * BoxX
    if (abs(DY) > BoxY / 2):
        DY = DY - round(RatioY) * BoxY
    if (abs(DZ) > BoxZ / 2):
        DZ = DZ - round(RatioZ) * BoxZ

    Distance = math.sqrt((DX) ** 2 + (DY) ** 2 + (DZ) ** 2)

    # print("Distance is: " + str(Distance) + " and RhoValue is: " + str(RhoValue))
    return Distance, DX, DY, DZ
# </editor-fold>

# <editor-fold desc="LogExplorer">
def LogExplorer(LogFileAddress_V, Type_V, Along_V, Rev_V , RhoFinder_V,MirrorAtEq_V,SameDirectory_V):
    # print(LogFileAddress_V)
    # os.system("pause")
    LogFile = open(LogFileAddress_V, "r")
    # print(LogFile)
    LogContent = LogFile.read()
    # print(LogContent)
    LogContent_split = LogContent.split()
    # print(LogContent_split)
    LogContent_split_reverse = LogContent_split[::-1]
    # print("Lattice Size Ratio: " + str(root.split("\\")[-1][3:]))
    # print("Size of Lattice parameter is " + str(DumpLatticeSize))
    # FileName = LogFileAddress_V.split("/")[-1]  # lattice information from file name
    FileName = ntpath.basename(LogFileAddress_V)  # lattice information from file name
    # print(FileName)

    if "wall" in LogContent_split:  # Judge if the job has been computed normally
        # print("Output File opened: " + str(root.split("\\")[-1])[3:])

        # Collecting Basic Info
        if SameDirectory_V:
            FileNameBase = FileName.split(".")[0]
            if "_" in FileNameBase:
                Iteration = float(FileNameBase.split("_")[-1])
            else:
                Iteration = float(FileNameBase)
        else:
            FolderName = root.split("\\")[-1]
            if "_" in FolderName:
                Iteration = float(FolderName.split("_")[-1])
            else:
                Iteration = float(FolderName)
        # print("Iteration is: " + str(Iteration))
        # print("Size of Lattice parameter is " + str(LatticeSize))
        NumOfAtomsIndex = LogContent_split.index("Nlocal:") + 1
        NumOfAtoms = int(float(LogContent_split[NumOfAtomsIndex]))
        # print(NumOfAtoms)
        if NumOfAtoms ==0:
            print("Amir: Zero atoms in Log File")
            return False
        # print("For Lattise Size: " + str(OutputLatticeSize) + " Number of atoms/cells is: " + str(NumOfAtoms))

        # Collecting Box Info
        # BoxIndex = LogContent_split.index("create_box")  # Target the string "orthogonal" in the log file
        # print("For Lattise Size: " + str(LatticeSize) + " index of orthogonal is: " + str(BoxIndex))
        # BoxX = float(LogContent_split[BoxIndex + 11][1:])
        # BoxY = float(LogContent_split[BoxIndex + 12])
        # BoxZ = float(LogContent_split[BoxIndex + 13][:-1])

        # print("For Lattise Size: " + str(OutputLatticeSize) + " orthogonal box is: " + str(BoxX) + str(" ") + str(BoxY) + str(" ") + str(BoxZ))

        # Collecting Energies
        EnergyIndex = LogContent_split_reverse.index("TotEng")  # Target the energy information
        EvEnergy = float(LogContent_split_reverse[EnergyIndex - 10])
        # print(EvEnergy)
        # print("For LOG FILE: " + str(FileName) + " For Lattise Size: " + str(LatticeSize) + " Energy of EAM file: " + str(EvEnergy) + " Ry")

        if math.isnan(EvEnergy):
            print("Amir: Nan Error in Reading Energy; check the Log File")
            return False
        else:
            EnergyPerAtom = EvEnergy/NumOfAtoms
            # print(EnergyPerAtom)
            if EnergyPerAtom>1E8:
                print("Amir: Suspiciously large Energy-File skipped")
                return False

            else:
                if RhoFinder_V:
                    RhoIndex = LogContent_split_reverse.index('"${RhoFinish}"')  # Target the energy information
                    RhoValue = float(LogContent_split_reverse[RhoIndex - 1])
                    # print("For LOG FILE: " + str(FileName) + " For Lattise Size: " + str(LatticeSize) + " RhoValue of EAM file: " + str(RhoValue) + "  ")

                else:
                    RhoValue = 0

                if Type_V == "Qsd":
                    if Along_V == "AlongA":
                        BondLengthAlong_V = BondLengthAlongA
                        Rev_V = EdRevAlongA
                    elif Along_V == "AlongC":
                        BondLengthAlong_V = BondLengthAlongC
                        Rev_V = EdRevAlongC
                    elif Along_V == "AlongAc":
                        BondLengthAlong_V = BondLengthAlongAc
                        Rev_V = EdRevAlongAc
                    # print("Folder is: " + str(float(root.split("\\")[-1][3:])))
                    # print("Along_V is: " + str(Along_V))
                    # print("BondLengthAlong_V is: " + str(BondLengthAlong_V))
                    # print("EdRev_V is: " + str(EdRev_V))

                    BoxX = 0
                    BoxY = 0
                    BoxZ = 0

                    Distance = Iteration * Rev_V
                    # print("Distance is: " + str(Distance))
                    # creating atomic bond instead of distance
                    BondDistance = BondLengthAlong_V - Distance

                    if MirrorAtEq_V:
                        BondDistance = 2 * BondLengthAlong_V - BondDistance

                    # print("BondDistance is: " + str(BondDistance))
                    Length = BondDistance
                    LogResult = np.array([[Iteration, Length, BoxX, BoxY, BoxZ, NumOfAtoms, EvEnergy, EnergyPerAtom, RhoValue]])
                    # print(LogResult)

                elif Type_V == "Eos":
                    # Collecting Box Info
                    BoxIndex = LogContent_split.index("create_box")  # Target the string "orthogonal" in the log file
                    # print("For Lattise Size: " + str(LatticeSize) + " index of orthogonal is: " + str(BoxIndex))
                    BoxX = float(LogContent_split[BoxIndex + 11][1:])
                    BoxY = float(LogContent_split[BoxIndex + 12])
                    BoxZ = float(LogContent_split[BoxIndex + 13][:-1])
                    # print("For Lattise Size: " + str(OutputLatticeSize) + " orthogonal box is: " + str(BoxX) + str(" ") + str(BoxY) + str(" ") + str(BoxZ))

                    LatticeSize = Iteration * Rev_V + LatticeEquilibriumDist
                    Length = LatticeSize
                    LogResult = np.array([[Iteration, Length, BoxX, BoxY, BoxZ, NumOfAtoms, EvEnergy, EnergyPerAtom, RhoValue]])

                else:
                    print("No Mechanism selected")
                    return False

                # print(LogResult)
                return LogResult
    else:
        print("Amir: wall keyword not found for: " + str(FileName))
        return False


    LogFile.close()

# </editor-fold>

# <editor-fold desc="PotentialExplorer">
def PotentialExplorer(FileAddress_V,StageName_V,Plotting_V,Reporting_V):
    # <editor-fold desc="Reading">
    FileName = ntpath.basename(FileAddress_V)  # lattice information from file name
    # File = open(FileAddress_V, "r")
    with open(FileAddress_V) as File:
        Content = File.read()
        ContentLineless = Content.splitlines()
        # print(Content)
        Header = ContentLineless[:2]
        Line4 = ContentLineless[3]
        Line5 = ContentLineless[4]
        Line6 = ContentLineless[5]
        Body = ContentLineless[6:]
        # print(Body)
        Line4Split = Line4.split()
        Line5Split = Line5.split()
        Line6Split = Line6.split()
        BodySplit = "  ".join(Body).split(' ')
        BodySplit = list(filter(None, BodySplit))
        # print(BodySplit)
    # </editor-fold>

    # <editor-fold desc="Analysis">
    if Reporting_V:
        print("Header of Potential is : " + str(Header))

    ElementNo = Line4[0]
    ElementName = Line4[1]

    # In potential file, the 5th line is: Nrho, drho, Nr, dr, Cutoff
    Nrho = int(Line5Split[0])  # number of tabulated values for embedding function F(rho)
    drho = float(Line5Split[1]) # spacing in density
    Nr = int(Line5Split[2]) # number of tabulated values for effective charge function Z(r)
    dr = float(Line5Split[3])  # distance space for pair interaction and density in Angstrom
    Cutoff = float(Line5Split[4])  # Cut off
    Cutoff2 = round(dr * Nr, 6)  # 7.6E+000

    if Reporting_V:
        print("Nrho is : " + str(Nrho))
        print("drho is : " + str(drho))
        print("Nr is : " + str(Nr))
        print("dr is : " + str(dr))
        print("Cutoff is : " + str(Cutoff))
        print("Cutoff is : " + str(Cutoff2))

    #In potential file, the 6th line is: atomic number, mass, lattice constant, lattice type (e.g. FCC)
    AtomicNumber = int(Line6Split[0])  # number of tabulated values for embedding function F(rho)
    Mass = float(Line6Split[1]) # spacing in density
    LatticeConstant = float(Line6Split[2]) # number of tabulated values for effective charge function Z(r)
    Structure = Line6Split[3] # number of tabulated values for effective charge function Z(r)

    if Reporting_V:
        print("AtomicNumber is : " + str(AtomicNumber))
        print("Mass is : " + str(Mass))
        print("LatticeConstant is : " + str(LatticeConstant))
        print("Structure is : " + str(Structure))

    LastRho = Nrho * drho
    LastDist = Nr * dr
    PhysicalDist = np.mgrid[0:LastDist:dr]
    RhoDist = np.mgrid[0:LastRho:drho]
    # print(RhoDist)

    FList = BodySplit[:Nrho]
    EffectiveChargeList = BodySplit[Nrho + Nr:] #The units for the effective charge Z are “atomic charge” or sqrt(Hartree * Bohr-radii). This is used by LAMMPS to compute the pair potential term in the EAM energy expression as r*phi
    RhoList = BodySplit[Nrho:Nrho+Nr]
    # print(FList)

    FListShape = len(FList)
    EffectiveChargeListShape = len(EffectiveChargeList)
    RhoListShape = len(RhoList)
    # print(FListShape)
    # print(EffectiveChargeListShape)
    # print(RhoListShape)

    FNumpy = np.array(FList,dtype=float)
    EffectiveChargeNumpy = np.array(EffectiveChargeList,dtype=float)
    RhoNumpy = np.array(RhoList,dtype=float)
    # print(FNumpy)
    # print(EffectiveChargeNumpy)
    # print(RhoNumpy)
    # print(np.shape(FNumpy))
    # print(np.shape(EffectiveChargeNumpy))
    # print(np.shape(RhoNumpy))

    Fdf = pd.DataFrame({'Rho': RhoDist,'F': FNumpy})
    EffectiveChargedf = pd.DataFrame({'Dist': PhysicalDist,'EffectiveCharge': EffectiveChargeNumpy})
    Rhodf = pd.DataFrame({'Dist': PhysicalDist,'Rho': RhoNumpy})
    # print(Fdf)
    # print(EffectiveChargedf)
    # print(Rhodf)
    # Plotting_V = True

    Phidf = EffectiveChargedf
    Phidf["Phi"] = Phidf["EffectiveCharge"]/Phidf["Dist"]
    # Plotting_V=True
    # </editor-fold>

    # <editor-fold desc="Plotting">
    if Plotting_V:
        Title = "F"
        plt.scatter(Fdf["Rho"], Fdf["F"], color="r")
        plt.xlabel("Rho")
        plt.ylabel("F (eV)")
        plt.title(Title)
        # plt.xlim(0, 8)
        # plt.ylim(-10, 500)
        # plt.yscale("log")
        plt.savefig(Date + "-PotentialExplorer-" + StageName_V + "-" + Title)
        if PlottingShow:
            plt.show()
        else:
            plt.close()

        Title = "EffectiveCharge"
        plt.scatter(EffectiveChargedf["Dist"], EffectiveChargedf["EffectiveCharge"], color="r")
        plt.xlabel(r"Distance " + "(" + "Å" + r")")
        plt.ylabel("Effective Charge (sqrt(Hartree * Bohr-radii))")
        plt.title(Title)
        plt.xlim(0, 8)
        plt.ylim(-10, 100)
        # plt.yscale("log")
        plt.savefig(Date + "-PotentialExplorer-" + StageName_V + "-" + Title)
        if PlottingShow:
            plt.show()
        else:
            plt.close()

        Title = "Rho"
        plt.scatter(Rhodf["Dist"], Rhodf["Rho"], color="r")
        plt.xlabel(r"Distance " + "(" + "Å" + r")")
        plt.ylabel("Rho")
        plt.title(Title)
        plt.xlim(0, 8)
        plt.ylim(-100, 10000)
        # plt.yscale("log")
        # plt.grid()
        plt.savefig(Date + "-PotentialExplorer-" + StageName_V + "-" + Title)
        if PlottingShow:
            plt.show()
        else:
            plt.close()

        Title = "Phi"
        plt.scatter(Phidf["Dist"], Phidf["Phi"], color="r")
        plt.xlabel(r"Distance " + "(" + "Å" + r")")
        plt.ylabel("Phi")
        plt.title(Title)
        plt.xlim(0, 8)
        plt.ylim(-100, 10000)
        # plt.yscale("log")
        # plt.grid()
        plt.savefig(Date + "-PotentialExplorer-" + StageName_V + "-" + Title)
        if PlottingShow:
            plt.show()
        else:
            plt.close()
    # </editor-fold>

    return Fdf,EffectiveChargedf,Rhodf,Phidf
# </editor-fold>

# <editor-fold desc="LammpsReportExplorer">
def LammpsReportExplorer(FileAddress_V, FunctionType_V , EdRev_V, BondLengthAlong_V, ReadIterationFrom_V, Iteration_V, MirrorAtEq_V, MeasureRho_V,SkipHeader_V):
    # print(FileAddress_V)

    # <editor-fold desc="File Info">
    FolderPath = os.path.dirname(FileAddress_V)
    FolderName = os.path.basename(FolderPath)
    FileName = ntpath.basename(FileAddress_V)
    FileNameBase = Path(FileAddress_V).stem
    # print(FolderPath)
    # print(FolderName)
    # print(FileName)
    # print(FileNameBase)
    if ReadIterationFrom_V == "FileName":
        if "_" in FileNameBase:
            Iteration = float(FileNameBase.split("_")[-1])
        else:
            Iteration = float(FileNameBase)
    elif ReadIterationFrom_V == "FolderName":
        if "_" in FolderName:
            Iteration = float(FolderName.split("_")[-1])
        else:
            Iteration = float(FolderName)
    elif ReadIterationFrom_V == "TimesCalled":
            Iteration = Iteration_V

    # </editor-fold>

    if MeasureRho_V:
        Dtype = [('Step', '<i4'), ('Time', 'f8'), ('dt', 'f8'), ('Cpu', 'f8'), ('TemperatureIonic', 'f8'), ('Press', 'f8'),('EnergyPotential', 'f8'), ('EnergyKinetic', 'f8'), ('Rho', 'f8')]
    else:
        Dtype = [('Step', '<i4'), ('Time', 'f8'), ('dt', 'f8'), ('Cpu', 'f8'), ('TemperatureIonic', 'f8'), ('Press', 'f8'), ('EnergyPotential', 'f8'), ('EnergyKinetic', 'f8')]
    ReportNumpy = np.genfromtxt(FileAddress_V, delimiter=",", skip_header=SkipHeader_V,max_rows=1)  # skip_header=1,dtype=[('Step','i8'),('EnergyPotential','f8')]"i8,f8,f8,f8,f8,f8,f8,f8

    # print(ReportNumpy[0])
    # print(ReportNumpy[7])
    ReportNumpy = np.append(Iteration, ReportNumpy)
    ReportNumpy = np.array([ReportNumpy])

    # print(ReportNumpy)
    return ReportNumpy

# </editor-fold>

# <editor-fold desc="DumpExplorer">
def DumpExplorer(DumpFileAddress_V,BatchRead_V):
    # print("Dump Explorer")
    # print("DumpFileAddress is: " + str(DumpFileAddress_V))
    NameLatticeBoxConfigNumpy = np.zeros((0,9))

    DumpFile = open(DumpFileAddress_V , "r")
    DumpContent = DumpFile.read()
    DumpContent_split = DumpContent.split()
    DumpContent_split_reverse = DumpContent_split[::-1]
    # print("Lattice Size Ratio: " + str(root.split("\\")[-1][3:]))
    # print("Size of Lattice parameter is " + str(DumpLatticeSize))

    # Collecting Basic Info
    Target = Path(DumpFileAddress_V).name
    if BatchRead_V == "SameFolder":
        FileName = float(float(Target.split(".")[0]))
        Lattice = FileName * EosRev + LatticeEquilibriumDist
        Tally = FileName
    elif BatchRead_V == "Foldered":
        FolderName = root.split("\\")[-1]
        # print(FolderName)
        if FolderName.lstrip("-").replace(".","",1).isdigit():
            # print("is")
            FolderName = float(FolderName)
            Lattice = FolderName * EosRev + LatticeEquilibriumDist
            Tally = FolderName

        else:
            # print("isn"t")
            FolderName = 0
            Lattice = LatticeEquilibriumDist
            Tally = FolderName
    else:
        FileName = str(Target.split(".")[0])
        Lattice = 0
        Tally = 0

    # print("Size of Lattice parameter is " + str(DumpLatticeSize))
    NumOfAtomsIndex = DumpContent_split.index("NUMBER")
    NumOfAtoms = int(DumpContent_split[NumOfAtomsIndex + 3])

    # Collecting Config
    ConfigNumpy = np.genfromtxt(DumpFileAddress_V, delimiter=" ", skip_header=9)
    # print("The shape of ConfigNumpy is: " + str(np.shape(ConfigNumpy)))
    # print("length of ConfigNumpy is: " + str(len(ConfigNumpy)))
    # print(DumpConfigSingleNumpy)

    # Collecting Box Info
    SkipFooterBox = NumOfAtoms + 1
    BoxNumpy = np.genfromtxt(DumpFileAddress_V , delimiter=" ", skip_header=5, skip_footer=SkipFooterBox)
    # print("BoxNumpy is:\n" + str(BoxNumpy))
    BoxX1 = BoxNumpy[0][0]
    BoxX2 = BoxNumpy[0][1]
    BoxY1 = BoxNumpy[1][0]
    BoxY2 = BoxNumpy[1][1]
    BoxZ1 = BoxNumpy[2][0]
    BoxZ2 = BoxNumpy[2][1]
    # print(DumpBoxNumpy[0][1])
    NameLatticeBoxNumpy = np.ones((len(ConfigNumpy), 8))
    NameLatticeBoxNumpy[:, 0] = NameLatticeBoxNumpy[:, 0] * Tally
    NameLatticeBoxNumpy[:, 1] = NameLatticeBoxNumpy[:, 1] * Lattice
    NameLatticeBoxNumpy[:, 2] = NameLatticeBoxNumpy[:, 2] * BoxX1
    NameLatticeBoxNumpy[:, 3] = NameLatticeBoxNumpy[:, 3] * BoxX2
    NameLatticeBoxNumpy[:, 4] = NameLatticeBoxNumpy[:, 4] * BoxY1
    NameLatticeBoxNumpy[:, 5] = NameLatticeBoxNumpy[:, 5] * BoxY2
    NameLatticeBoxNumpy[:, 6] = NameLatticeBoxNumpy[:, 6] * BoxZ1
    NameLatticeBoxNumpy[:, 7] = NameLatticeBoxNumpy[:, 7] * BoxZ2

    # print("The shape of NameLatticeBoxConfigNumpy is: " + str(np.shape(NameLatticeBoxConfigNumpy)))
    NameLatticeBoxConfigNumpy = np.append(NameLatticeBoxNumpy, ConfigNumpy, axis=-1)
    # NameLatticeBoxConfigNumpy is: FileName,Lattice,BoxX1,BoxX2,BoxY1,BoxY2,BoxZ1,BoxZ2,id,type,x,y,z,c_eng

    # DumpConfigSingleNumpyName = Date + "-DumpConfigSingleNumpy" + str(DumpLatticeSize) + ".csv"
    # np.savetxt(DumpConfigSingleNumpyName, DumpConfigSingleNumpy, delimiter=",",header="Lattice,AtomID,Type,X,Y,Z,E_Total", comments="")
    # print("The shape of NameLatticeBoxConfigNumpy is: " + str(np.shape(NameLatticeBoxConfigNumpy)))
    # print(DumpConfigSingleNumpy)

    DumpFile.close()

    return NameLatticeBoxConfigNumpy
# </editor-fold>

# <editor-fold desc="DftRelaxExplorer">
def DftRelaxExplorer(DftFileAddress_V, BondLenght_V, EmbeddedDimer_V = False, EmbeddedAtomNumber_V = 0, EOS_V = False):
    # print("Dump Explorer")
    DftEnergyResult = np.zeros((0, 5))
    NameLatticeBoxConfigDimerDistance = np.zeros((0, 13))
    NameLatticeBoxConfigNumpy = np.zeros((0,9))

    DftFile = open(DftFileAddress_V , "r")
    DftContent = DftFile.read()
    DftContent_split = DftContent.split()
    DftContent_split_reverse = DftContent_split[::-1]
    # print("Lattice Size Ratio: " + str(root.split("\\")[-1][3:]))
    # print("Size of Lattice parameter is " + str(DumpLatticeSize))
    if "DONE." in DftContent_split:

        # __________________________________________________Collecting General Info
        # print(root.split("\\")[-1])
        FileName = float(root.split("\\")[-1])
        # print("Size of Lattice parameter is " + str(DumpLatticeSize))
        ExclamationMarkIndex = DftContent_split.index("!")
        DftEnergyRy = float(DftContent_split[ExclamationMarkIndex + 4])

        AtomPerCellIndex = DftContent_split.index("atoms/cell")
        NumberOfAtoms = int(DftContent_split[AtomPerCellIndex + 2])

        DftEnergyEv = DftEnergyRy * Ry2eV
        DftEnergyEvPerAtom = DftEnergyRy * Ry2eV / NumberOfAtoms
        DftEnergyEvTotal = (DftEnergyRy + IsolatedAtomEv * NumberOfAtoms) * Ry2eV * 250

        # ___________________________________________________Collecting Config

        # Number of total Lines in File
        TotalLines = 0
        for line in DftFile:
            line = line.strip("\n")
            TotalLines += 1

        AtomPositionStartLine = 0
        with open(DftFileAddress_V) as DftFile:
            for number, line in enumerate(DftFile, 1):
                if "ATOMIC_POSITIONS (angstrom)" in line:
                    AtomPositionStartLine = number
                    # print("AtomPositionStartLine is: " + str(AtomPositionStartLine))

        ConfigNumpy = np.genfromtxt(DftFileAddress_V, delimiter="  ", skip_header=AtomPositionStartLine, max_rows=NumberOfAtoms, usecols=(3, 4, 5), dtype=float)

        # ___________________________________________________Collecting Box
        BoxA1Index = DftContent_split.index("a(1)")
        # print("A1X is: " + str(DftContent_split[BoxA1Index+3]))
        BoxA1 = float(DftContent_split[BoxA1Index + 3])
        BoxA2Index = DftContent_split.index("a(2)")
        # print("A2Y is: " + str(DftContent_split[BoxA2Index + 4]))
        BoxA2 = float(DftContent_split[BoxA2Index + 4])
        BoxA3Index = DftContent_split.index("a(3)")
        # print("A3Z is: " + str(DftContent_split[BoxA3Index + 5]))
        BoxA3 = float(DftContent_split[BoxA3Index + 5])


        # ___________________________________________________Mixing data
        infoNumpy = np.ones((len(ConfigNumpy), 9))
        infoNumpy[:, 0] = infoNumpy[:, 0] * FileName
        infoNumpy[:, 1] = infoNumpy[:, 1] * NumberOfAtoms
        infoNumpy[:, 2] = infoNumpy[:, 2] * DftEnergyEv
        infoNumpy[:, 3] = infoNumpy[:, 3] * BoxA1
        infoNumpy[:, 4] = infoNumpy[:, 4] * 0
        infoNumpy[:, 5] = infoNumpy[:, 5] * BoxA2
        infoNumpy[:, 6] = infoNumpy[:, 6] * 0
        infoNumpy[:, 7] = infoNumpy[:, 7] * BoxA3
        infoNumpy[:, 8] = infoNumpy[:, 8] * 0

        # Result is [[FileName, NumberOfAtoms, DftEnergyEv, BoxA1, 0, BoxA2, 0 , BoxA3, 0, PosX, PosY, PosZ]]
        # print("The shape of NameLatticeBoxConfigNumpy is: " + str(np.shape(NameLatticeBoxConfigNumpy)))
        infoConfigNumpy = np.append(infoNumpy, ConfigNumpy, axis=-1)


        if EmbeddedDimer_V:
            if infoConfigNumpy.size > 0:
                # print(DftNew[[17]])
                DimerDistance = BondLenght_V - (1/BoxDftToEamX) * ((infoConfigNumpy[EmbeddedAtomNumber_V, 9] - InitialPosX)**2 +
                                                            (infoConfigNumpy[EmbeddedAtomNumber_V, 10] - InitialPosY)**2 +
                                                            (infoConfigNumpy[EmbeddedAtomNumber_V, 11] - InitialPosZ)**2)**0.5
                # print("Distance is: " + str(Distance))
                infoConfigDimerDistance = np.append(infoConfigNumpy[[EmbeddedAtomNumber_V]],np.array([[DimerDistance]]), axis=1)
                # infoConfigDimerDistance is [[FileName, NumberOfAtoms, DftEnergyEv, BoxA1, 0, BoxA2, 0 , BoxA3, 0, PosX, PosY, PosZ , Distance]]
                # print("New Dimer DFT is: " + str(infoConfigDimerDistance))
                return infoConfigDimerDistance


        elif EOS_V:
            if infoConfigNumpy.size > 0:
                LatticeDistance = float(root.split("\\")[-1]) * EosRev + LatticeEquilibriumDist
                DftEos = np.array([[FileName, LatticeDistance, NumberOfAtoms, DftEnergyEv, DftEnergyEvPerAtom]])
                # infoConfigDimerDistance is [[FileName, NumberOfAtoms, DftEnergyEv, BoxA1, 0, BoxA2, 0 , BoxA3, 0, PosX, PosY, PosZ , Distance]]
                # print("New EOS DFT is: " + str(infoConfigEosDist))
                return DftEos
        else:
            print("Something is wrong!")
            return False



    DftFile.close()

# </editor-fold>

# <editor-fold desc="DftScfExplorer">
def DftScfExplorer(DftFileAddress_V, BondLenght_V, EmbeddedDimer_V = False, EmbeddedAtomNumber_V = 0, EOS_V = False):
    # print("DftScfExplorer")
    EnergyResult = np.zeros((0, 5))
    infoConfigNumpy = np.zeros((0, 12))
    infoConfigDimerDist = np.zeros((0, 5))

    # print(DftFileAddress_V)
    DftFile = open(DftFileAddress_V , "r")
    DftContent = DftFile.read()
    DftContent_split = DftContent.split()
    DftContent_split_reverse = DftContent_split[::-1]
    # print("Lattice Size Ratio: " + str(root.split("\\")[-1][3:]))
    # print("Size of Lattice parameter is " + str(DumpLatticeSize))
    if "DONE." in DftContent_split:

        # __________________________________________________Collecting General Info
        # print("DFT File opened is: " + str(root.split("\\")[-1]))
        FileName = float(root.split("\\")[-1])
        ExclamationMarkIndex = DftContent_split.index("!")
        DftEnergyRy = float(DftContent_split[ExclamationMarkIndex + 4])

        # DftEnergyEv = DftEnergyRy * Ry2eV

        AtomPerCellIndex = DftContent_split.index("atoms/cell")
        NumberOfAtoms = int(DftContent_split[AtomPerCellIndex + 2])

        DftEnergyEv = DftEnergyRy * Ry2eV

        DftEnergyEvPerAtom = DftEnergyRy * Ry2eV / NumberOfAtoms

        # ___________________________________________________Collecting Config

        #Number of total Lines in File
        TotalLines = 0
        for line in DftFile:
            line = line.strip("\n")
            TotalLines += 1


        AtomPositionStartLine = 0
        with open(DftFileAddress_V) as DftFile:
            for number, line in enumerate(DftFile, 1):
                if "positions" in line:
                    AtomPositionStartLine = number
                    # print("AtomPositionStartLine is: " + str(AtomPositionStartLine))


        ConfigNumpy = np.genfromtxt(DftFileAddress_V, delimiter="  ", skip_header=AtomPositionStartLine, max_rows=NumberOfAtoms, usecols=(8, 9, 10), dtype=float)
        # print("ConfigNumpy for 16th atom is: " + str(ConfigNumpy[16]))

        # ___________________________________________________Collecting Box
        BoxA1Index = DftContent_split.index("a(1)")
        # print("A1X is: " + str(DftContent_split[BoxA1Index+3]))
        BoxA1 = float(DftContent_split[BoxA1Index + 3])
        BoxA2Index = DftContent_split.index("a(2)")
        # print("A2Y is: " + str(DftContent_split[BoxA2Index + 4]))
        BoxA2 = float(DftContent_split[BoxA2Index + 4])
        BoxA3Index = DftContent_split.index("a(3)")
        # print("A3Z is: " + str(DftContent_split[BoxA3Index + 5]))
        BoxA3 = float(DftContent_split[BoxA3Index + 5])

        # ___________________________________________________Mixing data
        infoNumpy = np.ones((len(ConfigNumpy), 10))
        infoNumpy[:, 0] = infoNumpy[:, 0] * FileName
        infoNumpy[:, 1] = infoNumpy[:, 1] * NumberOfAtoms
        infoNumpy[:, 2] = infoNumpy[:, 2] * DftEnergyEv
        infoNumpy[:, 3] = infoNumpy[:, 2] * DftEnergyEvPerAtom
        infoNumpy[:, 4] = infoNumpy[:, 3] * BoxA1
        infoNumpy[:, 5] = infoNumpy[:, 4] * 0
        infoNumpy[:, 6] = infoNumpy[:, 5] * BoxA2
        infoNumpy[:, 7] = infoNumpy[:, 6] * 0
        infoNumpy[:, 8] = infoNumpy[:, 7] * BoxA3
        infoNumpy[:, 9] = infoNumpy[:, 8] * 0

        # Result is [[FileName, NumberOfAtoms, DftEnergyEv, BoxA1, 0, BoxA2, 0 , BoxA3, 0, PosX, PosY, PosZ]]
        # print("The shape of NameLatticeBoxConfigNumpy is: " + str(np.shape(NameLatticeBoxConfigNumpy)))
        infoConfigNumpy = np.append(infoNumpy, ConfigNumpy, axis=-1)

        if EmbeddedDimer_V:
            if infoConfigNumpy.size > 0:
                # print(DftNew[[17]])
                DimerDistance = BondLenght_V - (1/BoxDftToEamX) * ((infoConfigNumpy[EmbeddedAtomNumber_V, 10] - InitialPosX)**2 +
                                                            (infoConfigNumpy[EmbeddedAtomNumber_V, 11] - InitialPosY)**2 +
                                                            (infoConfigNumpy[EmbeddedAtomNumber_V, 12] - InitialPosZ)**2)**0.5
                # print("Distance is: " + str(Distance))
                DftEd = np.array([[FileName, DimerDistance, NumberOfAtoms, DftEnergyEv, DftEnergyEvPerAtom]])
                # print("New Dimer DFT is: " + str(infoConfigDimerDistance))
                return DftEd


        elif EOS_V:
            if infoConfigNumpy.size > 0:
                LatticeDistance = float(root.split("\\")[-1]) * EosRev + LatticeEquilibriumDist
                DftEos = np.array([[FileName, LatticeDistance, NumberOfAtoms, DftEnergyEv, DftEnergyEvPerAtom]])
                # infoConfigDimerDistance is [[FileName, NumberOfAtoms, DftEnergyEv, BoxA1, 0, BoxA2, 0 , BoxA3, 0, PosX, PosY, PosZ , Distance]]
                # print("New EOS DFT is: " + str(infoConfigEosDist))
                return DftEos
        else:
            print("Something is wrong!")
            return False
    else:
        return infoConfigDimerDist
    DftFile.close()

# </editor-fold>

# <editor-fold desc="RhoCalculator">
def RhoCalculator(ConfigNumpy_V,CutOff_V, BondLengthAlong_V, DistRhoSpline_V, EmbeddedDimer_V = False, EmbeddedAtomNumber_V = 16,EOS_V = False):
    RhoNumpy = np.zeros((0,4))
    AtomCounter = 0
    # print("Length of Numpy is: " + str(len(ConfigNumpy_V)))
    # print("ConfigNumpy_V is: " + str(ConfigNumpy_V))
    #
    # print("CalculatorPpm is: " + str(CalculatorPpm))
    # print("CalculatorMax is: " + str(CalculatorMax))

    if EmbeddedDimer_V:
        # print(0)

        for Atom in range(len(ConfigNumpy_V)): #range(CalculatorPpm,CalculatorMax):
            # print("Atom is: " + str(Atom))
            Rho = 0

            FileName = ConfigNumpy_V[Atom][0]
            Lattice = ConfigNumpy_V[Atom][1]
            BoxX1 = ConfigNumpy_V[Atom][2]
            BoxX2 = ConfigNumpy_V[Atom][3]
            BoxY1 = ConfigNumpy_V[Atom][4]
            BoxY2 = ConfigNumpy_V[Atom][5]
            BoxZ1 = ConfigNumpy_V[Atom][6]
            BoxZ2 = ConfigNumpy_V[Atom][7]
            AtomID = ConfigNumpy_V[Atom][8]
            AtomType = ConfigNumpy_V[Atom][9]
            AtomX = ConfigNumpy_V[Atom][10]
            AtomY = ConfigNumpy_V[Atom][11]
            AtomZ = ConfigNumpy_V[Atom][12]
            AtomC_Energy = ConfigNumpy_V[Atom][13]

            BoxX = BoxX2 - BoxX1
            BoxY = BoxY2 - BoxY1
            BoxZ = BoxZ2 - BoxZ1


            MovingAtomID = ConfigNumpy_V[EmbeddedAtomNumber_V][8]
            MovingAtomX = ConfigNumpy_V[EmbeddedAtomNumber_V][10]
            MovingAtomY = ConfigNumpy_V[EmbeddedAtomNumber_V][11]
            MovingAtomZ = ConfigNumpy_V[EmbeddedAtomNumber_V][12]


            # print("MovingAtomID: " + str(MovingAtomID) + " MovingAtomX: " + str(MovingAtomX) + " MovingAtomY: " + str(MovingAtomY) + " EmbeddedAtomZ: " + str(MovingAtomZ))
            # print("X: " + str(AtomX) + " Y: " + str(AtomY) + " Z: " + str(AtomZ))

            EmbeddedAtomPpmInitialAtomDist = PBCDist(MovingAtomX, MovingAtomY, MovingAtomZ, InitialPosX, InitialPosY, InitialPosZ, BoxX, BoxY, BoxZ)[0]  # Calculates the distance for neighbors!
            DimerDistance = BondLengthAlong_V - EmbeddedAtomPpmInitialAtomDist

            for Opponent in range(len(ConfigNumpy_V)):
                # print("Opponant is: " + str(Opponent))
                OpponentID = ConfigNumpy_V[Opponent][8]
                OpponentX = ConfigNumpy_V[Opponent][10]
                OpponentY = ConfigNumpy_V[Opponent][11]
                OpponentZ = ConfigNumpy_V[Opponent][12]

                # Dist
                NewPBCDist = PBCDist(AtomX, AtomY, AtomZ, OpponentX, OpponentY, OpponentZ, BoxX, BoxY, BoxZ)[0]  # Calculates the distance for neighbors!
                # print("Distance is: " + str(NewPBCDist))
                # print("Main Atom is: " + str(Atom) + " and its core status is: " + str(AtomCore) + " -> Opponent Atom is: " + str(Opponent) + " and its core status is: " + str(OpponentCore))

                if Atom == Opponent:
                    continue
                else:

                    if NewPBCDist > CutOff_V:
                        continue
                    else:
                        # print("Within Cut Off")
                        # Rho
                        PhiNew = DistRhoSpline_V(NewPBCDist)
                        Rho = Rho + PhiNew
            NewRow = [[FileName,DimerDistance,AtomID,Rho]]
            # print("EmbeddedDimer")
            # print("NewRow in RhoCalculator is:\n" + str(NewRow))
            # print("RhoNumpy in RhoCalculator Before adding a New One is:\n" + str(RhoNumpy))
            RhoNumpy = np.append(RhoNumpy,NewRow,axis=0)
            # print("RhoNumpy in RhoCalculator After adding a New One is:\n" + str(RhoNumpy))
        RhoSum = RhoNumpy.sum(axis=0)[3]
        # print("RhoSum in RhoCalculator is:\n" + str(RhoSum))
        RhoAve = RhoNumpy.mean(axis=0)[3]
        # print("RhoNumpy is:\n" + str(RhoNumpy))
        return RhoNumpy, RhoSum, RhoAve

    elif EOS_V:
        # print(1)

        for Atom in range(CalculatorMin,CalculatorMax):
            # print("Atom is: " + str(Atom))
            Rho = 0

            FileName = ConfigNumpy_V[Atom][0]
            Lattice = ConfigNumpy_V[Atom][1]
            BoxX1 = ConfigNumpy_V[Atom][2]
            BoxX2 = ConfigNumpy_V[Atom][3]
            BoxY1 = ConfigNumpy_V[Atom][4]
            BoxY2 = ConfigNumpy_V[Atom][5]
            BoxZ1 = ConfigNumpy_V[Atom][6]
            BoxZ2 = ConfigNumpy_V[Atom][7]
            AtomID = ConfigNumpy_V[Atom][8]
            AtomType = ConfigNumpy_V[Atom][9]
            AtomX = ConfigNumpy_V[Atom][10]
            AtomY = ConfigNumpy_V[Atom][11]
            AtomZ = ConfigNumpy_V[Atom][12]
            AtomC_Energy = ConfigNumpy_V[Atom][13]

            BoxX = BoxX2 - BoxX1
            BoxY = BoxY2 - BoxY1
            BoxZ = BoxZ2 - BoxZ1

            for Opponent in range(len(ConfigNumpy_V)):
                # print("Opponant is: " + str(Opponent))
                OpponentID = ConfigNumpy_V[Opponent][8]
                OpponentX = ConfigNumpy_V[Opponent][10]
                OpponentY = ConfigNumpy_V[Opponent][11]
                OpponentZ = ConfigNumpy_V[Opponent][12]

                # Dist
                NewPBCDist = PBCDist(AtomX, AtomY, AtomZ, OpponentX, OpponentY, OpponentZ, BoxX, BoxY, BoxZ)[0]  # Calculates the distance for neighbors!
                # print("Distance is: " + str(NewPBCDist))
                # print("Main Atom is: " + str(Atom) + " and its core status is: " + str(AtomCore) + " -> Opponent Atom is: " + str(Opponent) + " and its core status is: " + str(OpponentCore))

                if Atom == Opponent:
                    continue
                else:

                    if NewPBCDist > CutOff_V:
                        continue
                    else:
                        # print("Within Cut Off")
                        # Rho
                        PhiNew = DistRhoSpline_V(NewPBCDist)
                        Rho = Rho + PhiNew
            NewRow = [[FileName,Lattice,AtomID,Rho]]
            # print("EOS")
            # print("NewRow in RhoCalculator is:\n" + str(NewRow))
            # print("RhoNumpy in RhoCalculator Before adding a New One is:\n" + str(RhoNumpy))
            RhoNumpy = np.append(RhoNumpy,NewRow,axis=0)
            # print("RhoNumpy in RhoCalculator After adding a New One is:\n" + str(RhoNumpy))
        RhoSum = RhoNumpy.sum(axis=0)[3]
        # print("RhoSum in RhoCalculator is:\n" + str(RhoSum))
        RhoAve = RhoNumpy.mean(axis=0)[3]
        # print("RhoAve is:\n" + str(RhoAve))
        # print(RhoNumpy)
        return RhoNumpy, RhoSum, RhoAve
    else:
        print("Something is wrong with RhoCalculator!")
        return False

# </editor-fold>

# <editor-fold desc="VSumCalculator">
def VSumCalculator(ConfigNumpy_V,CutOff_V, BondLengthAlong_V, DistVSpline_V, EmbeddedDimer_V = False, EmbeddedAtomNumber_V = 16, EOS_V = False):
    AtomCounter = 0
    # print("Length of Numpy is: " + str(len(ConfigNumpy_V)))
    VNumpy = np.zeros((0,4))
    VSumResult = np.array([[0, 0, 0]])

    if EmbeddedDimer_V:
        for Atom in range(len(ConfigNumpy_V)):  # range(CalculatorPpm,CalculatorMax):

            # print("Atom is: " + str(Atom))
            VAtomSum = 0

            FileName = ConfigNumpy_V[Atom][0]
            Lattice = ConfigNumpy_V[Atom][1]
            BoxX1 = ConfigNumpy_V[Atom][2]
            BoxX2 = ConfigNumpy_V[Atom][3]
            BoxY1 = ConfigNumpy_V[Atom][4]
            BoxY2 = ConfigNumpy_V[Atom][5]
            BoxZ1 = ConfigNumpy_V[Atom][6]
            BoxZ2 = ConfigNumpy_V[Atom][7]
            AtomID = ConfigNumpy_V[Atom][8]
            AtomType = ConfigNumpy_V[Atom][9]
            AtomX = ConfigNumpy_V[Atom][10]
            AtomY = ConfigNumpy_V[Atom][11]
            AtomZ = ConfigNumpy_V[Atom][12]
            AtomC_Energy = ConfigNumpy_V[Atom][13]

            BoxX = BoxX2 - BoxX1
            BoxY = BoxY2 - BoxY1
            BoxZ = BoxZ2 - BoxZ1

            MovingAtomID = ConfigNumpy_V[EmbeddedAtomNumber_V][8]
            MovingAtomX = ConfigNumpy_V[EmbeddedAtomNumber_V][10]
            MovingAtomY = ConfigNumpy_V[EmbeddedAtomNumber_V][11]
            MovingAtomZ = ConfigNumpy_V[EmbeddedAtomNumber_V][12]


            # print("MovingAtomID: " + str(MovingAtomID) + " MovingAtomX: " + str(MovingAtomX) + " MovingAtomY: " + str(MovingAtomY) + " EmbeddedAtomZ: " + str(MovingAtomZ))
            # print("X: " + str(AtomX) + " Y: " + str(AtomY) + " Z: " + str(AtomZ))

            EmbeddedAtoPpmitialAtomDist = PBCDist(MovingAtomX, MovingAtomY, MovingAtomZ, InitialPosX, InitialPosY, InitialPosZ, BoxX, BoxY, BoxZ)[0]  # Calculates the distance for neighbors!
            DimerDistance = BondLengthAlong_V - EmbeddedAtoPpmitialAtomDist


            # for Atom in range(len(ConfigNumpy)):
            for Opponent in range(len(ConfigNumpy_V)):
                OpponentID = ConfigNumpy_V[Opponent][8]
                OpponentX = ConfigNumpy_V[Opponent][10]
                OpponentY = ConfigNumpy_V[Opponent][11]
                OpponentZ = ConfigNumpy_V[Opponent][12]

                # Dist
                NewPBCDist = PBCDist(AtomX, AtomY, AtomZ, OpponentX, OpponentY, OpponentZ, BoxX, BoxY, BoxZ)[0]  # Calculates the distance for neighbors!
                # print("Distance is: " + str(NewPBCDist))
                # print("Main Atom is: " + str(Atom) + " and its core status is: " + str(AtomCore) + " -> Opponent Atom is: " + str(Opponent) + " and its core status is: " + str(OpponentCore))

                if Atom == Opponent:
                    continue
                else:

                    if NewPBCDist > CutOff_V:
                        continue
                    else:
                        # print("Distance is: " + str(NewPBCDist))
                        # print("Within Cut Off")
                        VAtomNew = DistVSpline_V(NewPBCDist)
                        VAtomSum = VAtomSum + VAtomNew

            NewRow = [[FileName, DimerDistance, AtomID, VAtomSum]]
            # print(NewRow)
            VNumpy = np.append(VNumpy, NewRow, axis=0)

        FileName = VNumpy.mean(axis=0)[0]
        EmbeddedAtoPpmitialAtomDist = VNumpy.mean(axis=0)[1]
        VSum = VNumpy.sum(axis=0)[3]
        VSumResult = [[FileName, EmbeddedAtoPpmitialAtomDist, VSum]]

    elif EOS_V:
        for Atom in range(CalculatorMin,CalculatorMax):
            # print("Atom is: " + str(Atom))
            VAtomSum = 0

            FileName = ConfigNumpy_V[Atom][0]
            Lattice = ConfigNumpy_V[Atom][1]
            BoxX1 = ConfigNumpy_V[Atom][2]
            BoxX2 = ConfigNumpy_V[Atom][3]
            BoxY1 = ConfigNumpy_V[Atom][4]
            BoxY2 = ConfigNumpy_V[Atom][5]
            BoxZ1 = ConfigNumpy_V[Atom][6]
            BoxZ2 = ConfigNumpy_V[Atom][7]
            AtomID = ConfigNumpy_V[Atom][8]
            AtomType = ConfigNumpy_V[Atom][9]
            AtomX = ConfigNumpy_V[Atom][10]
            AtomY = ConfigNumpy_V[Atom][11]
            AtomZ = ConfigNumpy_V[Atom][12]
            AtomC_Energy = ConfigNumpy_V[Atom][13]

            BoxX = BoxX2 - BoxX1
            BoxY = BoxY2 - BoxY1
            BoxZ = BoxZ2 - BoxZ1

            # for Atom in range(len(ConfigNumpy)):
            for Opponent in range(len(ConfigNumpy_V)):
                OpponentID = ConfigNumpy_V[Opponent][8]
                OpponentX = ConfigNumpy_V[Opponent][10]
                OpponentY = ConfigNumpy_V[Opponent][11]
                OpponentZ = ConfigNumpy_V[Opponent][12]

                # Dist
                NewPBCDist = PBCDist(AtomX, AtomY, AtomZ, OpponentX, OpponentY, OpponentZ, BoxX, BoxY, BoxZ)[0]  # Calculates the distance for neighbors!
                # print("Distance is: " + str(NewPBCDist))
                # print("Main Atom is: " + str(Atom) + " and its core status is: " + str(AtomCore) + " -> Opponent Atom is: " + str(Opponent) + " and its core status is: " + str(OpponentCore))

                if Atom == Opponent:
                    continue
                else:

                    if NewPBCDist > CutOff_V:
                        continue
                    else:
                        # print("Distance is: " + str(NewPBCDist))
                        # print("Within Cut Off")
                        VAtomNew = DistVSpline_V(NewPBCDist)
                        VAtomSum = VAtomSum + VAtomNew

            NewRow = [[FileName, Lattice, AtomID, VAtomSum]]
            VNumpy = np.append(VNumpy, NewRow,axis=0)

        FileName = VNumpy.mean(axis=0)[0]
        LatticeSize = VNumpy.mean(axis=0)[1]
        VSum = VNumpy.sum(axis=0)[3]
        VSumResult = [[FileName, LatticeSize, VSum]]

    else:
        print("Something is wrong!")
        return False


    return VSumResult
# </editor-fold>

# <editor-fold desc="FCalculator">
def FCalculator(ConfigNumpy_V,CutOff_V,BondLengthAlong_V, DistPhiSpline_V, RhoFSpline_V, EmbeddedDimer_V = False, EmbeddedAtomNumber_V = 16, EOS_V = False):
    FNumpy = np.zeros((0,4))
    # print(1)
    RhoNumpy = RhoCalculator(ConfigNumpy_V, CutOff_V, BondLengthAlong_V, DistPhiSpline_V,EmbeddedDimer_V = EmbeddedDimer_V , EmbeddedAtomNumber_V = EmbeddedAtomNumber_V, EOS_V = EOS_V)[0]
    # print(2)
    # print(RhoNumpy)
    for atom in range(len(RhoNumpy)):
        FileName = RhoNumpy[atom][0]
        Lattice = RhoNumpy[atom][1]
        AtomID = RhoNumpy[atom][2]
        RhoAtom = RhoNumpy[atom][3]
        FNew = RhoFSpline_V(RhoAtom)
        RowNew = [[FileName,Lattice,AtomID,FNew]]
        FNumpy = np.append(FNumpy,RowNew,axis=0)

    # print("FNumpy is:\n" + str(FNumpy))
    FSum = FNumpy.sum(axis=0)[3]
    FAve = FNumpy.mean(axis=0)[3]

    return FNumpy,FSum,FAve
# </editor-fold>

# <editor-fold desc="EamCalculator">
def EamCalculator (DumpFileAddress_V, CutOff_V , DistPhiSpline_V, RhoFSpline_V, DistVSpline_V, BondLenght_V = BondLengthAlongA, EmbeddedDimer_V = False, EmbeddedAtomNumber_V = 16, EOS_V = False,MirrorAtEq_V=False,SameDirectory_V=False):
    # print("DumpFileAddress_V in EAM calculator is:\n" + str(DumpFileAddress_V))
    NameLatticeBoxConfigNumpy = DumpExplorer(DumpFileAddress_V, SameDirectory_V=SameDirectory_V)

    # NameLatticeBoxConfigNumpy is: FileName,Lattice,BoxX1,BoxX2,BoxY1,BoxY2,BoxZ1,BoxZ2,id,type,x,y,z,c_eng
    # print("The shape of DumpConfigNumpy is: " + str(np.shape(NameLatticeBoxConfigNumpy)))
    # print(NameLatticeBoxConfigNumpy)

    RhoResult = RhoCalculator(NameLatticeBoxConfigNumpy, CutOff_V, BondLenght_V, DistPhiSpline_V,EmbeddedDimer_V = EmbeddedDimer_V, EmbeddedAtomNumber_V = EmbeddedAtomNumber_V, EOS_V = EOS_V)
    # print("The shape of RhoOriginalSumNumpy is:\n" + str(np.shape(RhoOriginalNumpy[0])))
    # print("The RhoOriginalSum is:\n" + str(RhoResult[1]))
    # print("The RhoAverage is:\n" + str(RhoResult[2]))
    # print("The RhoOriginalSumNumpy is:\n" + str(RhoOriginalNumpy[1][3]))
    # print(1)
    FResult = FCalculator(NameLatticeBoxConfigNumpy, CutOff_V, BondLenght_V, DistPhiSpline_V,RhoFSpline_V,EmbeddedDimer_V = EmbeddedDimer_V, EmbeddedAtomNumber_V = 16, EOS_V = EOS_V)
    # print(2)
    # print("The shape of FOriginalNumpy is:\n" + str(np.shape(FOriginalNumpy[0])))
    # print("The FOriginalSum is:\n" + str(OriginalPotentialF[1]))
    # print("The FOriginalNumpy is:\n" + str(FOriginalNumpy[1][3]))
    # VSumCalculator input is (ConfigNumpy_V,CutOff_V, BondLengthAlong_V, DistVSpline_V, EmbeddedDimer_V = False, EmbeddedAtomNumber_V = 16, EOS_V = False):
    PhiResult = VSumCalculator(NameLatticeBoxConfigNumpy, CutOff_V, BondLenght_V, DistVSpline_V,EmbeddedDimer_V = EmbeddedDimer_V, EmbeddedAtomNumber_V = EmbeddedAtomNumber_V, EOS_V = EOS_V)
    # print(3)
    # print("The shape of PhiOriginalSumNumpy is:\n" + str(np.shape(PhiOriginalSumNumpy[0])))
    # print("The PhiResult is\n" + str(PhiResult))
    # print("The PhiOriginalSum is\n" + str(OriginalPotentialPhi[0][2]))

    # in this method, the properties of one single position was calculated and multipled by total number of atoms

    NumberOfAtomsInConfig = len(NameLatticeBoxConfigNumpy)
    # print("NumberOfAtomsInConfig is: " + str(NumberOfAtomsInConfig))
    NumberOfAtomsInCalculation = len(FResult[0])
    # print("NumberOfAtomsInCalculation is: " + str(NumberOfAtomsInCalculation))
    AmplitudeRatio = NumberOfAtomsInConfig/NumberOfAtomsInCalculation
    # print("AmplitudeRatio is: " + str(AmplitudeRatio))
    RhoCalculated = RhoResult[2]
    FCalculated = FResult[1]
    PhiCalculated = PhiResult[0][2]
    EnergyNew = FCalculated + 0.5 * PhiCalculated
    EnergyTotal = AmplitudeRatio * EnergyNew
    EnergyNewPerAtom = EnergyNew/NumberOfAtomsInCalculation
    # print(EnergyNewPerAtom)

    # print("The EnergyOriginalNumpy is:\n" + str(EnergyOriginalNumpy))
    FileNameNew = NameLatticeBoxConfigNumpy.mean(axis=0)[0]

    if EOS_V:
        # print("EamCalculator EOS")
        if NameLatticeBoxConfigNumpy.size > 0:
            # print(DftNew[[17]])
            LatticeSizeNew = NameLatticeBoxConfigNumpy.mean(axis=0)[1]
            # print(str(Distance) + " " + str(DimerDistance))
            # print("Distance is: " + str(Distance))
            # print("NameLatticeBoxConfigNumpy at: " + str(EmbeddedAtomNumber_V) + " is :\n" + str(NameLatticeBoxConfigNumpy[[EmbeddedAtomNumber_V]]))
            EamEosResult = np.append(NameLatticeBoxConfigNumpy[[0]], np.array([[LatticeSizeNew, RhoCalculated, FCalculated, PhiCalculated, EnergyTotal, EnergyNewPerAtom]]), axis=1)
            # EamDimerResult is: FileName,Lattice,BoxX1,BoxX2,BoxY1,BoxY2,BoxZ1,BoxZ2,id,type,x,y,z,c_eng,LatticeSizeNew,RhoCalculated,FCalculated,PhiCalculated,EnergyTotal,EnergyNewPerAtom
            # print("for " + str(NameLatticeBoxConfigNumpy.mean(axis=0)[0]) + " EamEosResult is:\n" + str(EamEosResult))
            EamResult = EamEosResult

    elif EmbeddedDimer_V:
        # print("EamCalculator EmbeddedDimer")
        if NameLatticeBoxConfigNumpy.size > 0:
            # print(DftNew[[17]])
            # print(EmbeddedAtomNumber_V)
            Distance = math.copysign(1,NameLatticeBoxConfigNumpy[EmbeddedAtomNumber_V,0]) *\
                       ((NameLatticeBoxConfigNumpy[EmbeddedAtomNumber_V, 10] - InitialPosX) ** 2 +
                        (NameLatticeBoxConfigNumpy[EmbeddedAtomNumber_V, 11] - InitialPosY) ** 2 +
                        (NameLatticeBoxConfigNumpy[EmbeddedAtomNumber_V, 12] - InitialPosZ) ** 2) ** 0.5
            # print(Distance)
            if FileNameNew>0:
                DimerDistance = BondLenght_V - abs(Distance)
            else:
                DimerDistance = BondLenght_V + abs(Distance)

            if MirrorAtEq_V:
                DimerDistance = 2*BondLenght_V-DimerDistance
            # print(str(FileNameNew)+ " " +str(EmbeddedAtomNumber_V) + " " + str(Distance) + " " + str(DimerDistance))
            # print("Distance is: " + str(Distance))
            # print("NameLatticeBoxConfigNumpy at: " + str(EmbeddedAtomNumber_V) + " is :\n" + str(NameLatticeBoxConfigNumpy[[EmbeddedAtomNumber_V]]))
            EamDimerResult = np.append(NameLatticeBoxConfigNumpy[[EmbeddedAtomNumber_V]], np.array([[DimerDistance, RhoCalculated, FCalculated, PhiCalculated, EnergyTotal, EnergyNewPerAtom]]), axis=1)
            # EamDimerResult is: FileName,Lattice,BoxX1,BoxX2,BoxY1,BoxY2,BoxZ1,BoxZ2,id,type,x,y,z,c_eng,DimerDistance, RhoCalculated, FCalculated, PhiCalculated, EnergyTotal, EnergyNewPerAtom
            # print("EamDimerResult is: " + str(EamDimerResult))
            EamResult =  EamDimerResult

    else:
        print("Something is wrong!")
        EamResult = False

    Result = [EamResult,RhoResult,FResult,PhiResult]
    return Result[0]
# </editor-fold>

# <editor-fold desc="ZBL">
def ZblSwitching(Ditance_V, R1, Rc):
    Result = 0
    return Result

def ZblPhi(X):
    Result = 0.18175 * math.exp(-3.19980 * X) + 0.50986 * math.exp(-0.94229 * X) + 0.28022 * math.exp(-0.40290 * X) + 0.02817 * math.exp(-0.20162 * X)
    return Result

def ZblE(Ditance_V):
    E_a = 0.46850 / (NuclearCharge**0.23 + NuclearCharge**0.23)
    E_fraction = 1 / (4 * const.pi * EpsilonNought)
    E_zbl = E_fraction * (NuclearCharge * NuclearCharge * ElectronCharge**2) * (ZblPhi(Ditance_V/E_a) / Ditance_V) + ZblSwitching(Ditance_V,R1, Rc)
    return E_zbl

InitialPosX = 0
InitialPosY = 0
InitialPosZ = 0

def ZblConfig(ConfigNumpy_V,CutOff_V,BondLengthAlong_V, EmbeddedDimer_V = False, EmbeddedAtomNumber_V = 16, EOS = False):

    ZblConfigNumpy = np.zeros((0, 4))
    np.array([[0, 0, 0]])
    AtomCounter = 0
    # print("Length of Numpy is: " + str(len(ConfigNumpy_V)))

    if EmbeddedDimer_V:

        for Atom in range(len(ConfigNumpy_V)):
            # print("Atom is: " + str(Atom))
            ZblAtom = 0

            FileName = ConfigNumpy_V[Atom][0]
            Lattice = ConfigNumpy_V[Atom][1]
            BoxX1 = ConfigNumpy_V[Atom][2]
            BoxX2 = ConfigNumpy_V[Atom][3]
            BoxY1 = ConfigNumpy_V[Atom][4]
            BoxY2 = ConfigNumpy_V[Atom][5]
            BoxZ1 = ConfigNumpy_V[Atom][6]
            BoxZ2 = ConfigNumpy_V[Atom][7]
            AtomID = ConfigNumpy_V[Atom][8]
            AtomType = ConfigNumpy_V[Atom][9]
            AtomX = ConfigNumpy_V[Atom][10]
            AtomY = ConfigNumpy_V[Atom][11]
            AtomZ = ConfigNumpy_V[Atom][12]
            AtomC_Energy = ConfigNumpy_V[Atom][13]

            BoxX = BoxX2 - BoxX1
            BoxY = BoxY2 - BoxY1
            BoxZ = BoxZ2 - BoxZ1

            MovingAtomID = ConfigNumpy_V[EmbeddedAtomNumber_V][8]
            MovingAtomX = ConfigNumpy_V[EmbeddedAtomNumber_V][10]
            MovingAtomY = ConfigNumpy_V[EmbeddedAtomNumber_V][11]
            MovingAtomZ = ConfigNumpy_V[EmbeddedAtomNumber_V][12]

            # print("MovingAtomID: " + str(MovingAtomID) + " MovingAtomX: " + str(MovingAtomX) + " MovingAtomY: " + str(MovingAtomY) + " EmbeddedAtomZ: " + str(MovingAtomZ))
            # print("X: " + str(AtomX) + " Y: " + str(AtomY) + " Z: " + str(AtomZ))

            EmbeddedAtoPpmitialAtomDist = PBCDist(MovingAtomX, MovingAtomY, MovingAtomZ, InitialPosX, InitialPosY, InitialPosZ, BoxX, BoxY, BoxZ)[0]  # Calculates the distance for neighbors!
            DimerDistance = BondLengthAlong_V - EmbeddedAtoPpmitialAtomDist


            for Opponent in range(len(ConfigNumpy_V)):
                # print("Opponant is: " + str(Opponent))
                OpponentID = ConfigNumpy_V[Opponent][8]
                OpponentX = ConfigNumpy_V[Opponent][10]
                OpponentY = ConfigNumpy_V[Opponent][11]
                OpponentZ = ConfigNumpy_V[Opponent][12]

                # Dist
                NewPBCDist = PBCDist(AtomX, AtomY, AtomZ, OpponentX, OpponentY, OpponentZ, BoxX, BoxY, BoxZ)[0]  # Calculates the distance for neighbors!
                # print("Distance is: " + str(NewPBCDist))
                # print("Main Atom is: " + str(Atom) + " and its core status is: " + str(AtomCore) + " -> Opponent Atom is: " + str(Opponent) + " and its core status is: " + str(OpponentCore))

                if Atom == Opponent:
                    continue
                else:

                    if NewPBCDist > CutOff_V:
                        continue
                    else:
                        # print("Within Cut Off")
                        # Rho
                        ZblAtomNew = ZblE(NewPBCDist)
                        ZblAtom = ZblAtom + ZblAtomNew

            NewRow = [[FileName, DimerDistance, AtomID, ZblAtom]]
            # print("NewRow in RhoCalculator is:\n" + str(NewRow))
            # print("RhoNumpy in RhoCalculator Before adding a New One is:\n" + str(RhoNumpy))
            ZblConfigNumpy = np.append(ZblConfigNumpy, NewRow, axis=0)
            # print("RhoNumpy in RhoCalculator After adding a New One is:\n" + str(RhoNumpy))

        FileName = ZblConfigNumpy.mean(axis=0)[0]
        EmbeddedAtoPpmitialAtomDist = ZblConfigNumpy.mean(axis=0)[1]
        ZblConfig = ZblConfigNumpy.sum(axis=0)[3]
        ZblResult = [[FileName, EmbeddedAtoPpmitialAtomDist, ZblConfig]]


    elif EOS:
        for Atom in range(len(ConfigNumpy_V)):
            # print("Atom is: " + str(Atom))
            ZBL = 0

            FileName = ConfigNumpy_V[Atom][0]
            Lattice = ConfigNumpy_V[Atom][1]
            BoxX1 = ConfigNumpy_V[Atom][2]
            BoxX2 = ConfigNumpy_V[Atom][3]
            BoxY1 = ConfigNumpy_V[Atom][4]
            BoxY2 = ConfigNumpy_V[Atom][5]
            BoxZ1 = ConfigNumpy_V[Atom][6]
            BoxZ2 = ConfigNumpy_V[Atom][7]
            AtomID = ConfigNumpy_V[Atom][8]
            AtomType = ConfigNumpy_V[Atom][9]
            AtomX = ConfigNumpy_V[Atom][10]
            AtomY = ConfigNumpy_V[Atom][11]
            AtomZ = ConfigNumpy_V[Atom][12]
            AtomC_Energy = ConfigNumpy_V[Atom][13]

            BoxX = BoxX2 - BoxX1
            BoxY = BoxY2 - BoxY1
            BoxZ = BoxZ2 - BoxZ1

            for Opponent in range(len(ConfigNumpy_V)):
                # print("Opponant is: " + str(Opponent))
                OpponentID = ConfigNumpy_V[Opponent][8]
                OpponentX = ConfigNumpy_V[Opponent][10]
                OpponentY = ConfigNumpy_V[Opponent][11]
                OpponentZ = ConfigNumpy_V[Opponent][12]

                # Dist
                NewPBCDist = PBCDist(AtomX, AtomY, AtomZ, OpponentX, OpponentY, OpponentZ, BoxX, BoxY, BoxZ)[0]  # Calculates the distance for neighbors!
                # print("Distance is: " + str(NewPBCDist))
                # print("Main Atom is: " + str(Atom) + " and its core status is: " + str(AtomCore) + " -> Opponent Atom is: " + str(Opponent) + " and its core status is: " + str(OpponentCore))

                if Atom == Opponent:
                    continue
                else:

                    if NewPBCDist > CutOff_V:
                        continue
                    else:
                        # print("Within Cut Off")
                        # Rho
                        ZBLNew = ZblE(NewPBCDist)
                        ZBL = ZBL + ZBLNew

            NewRow = [[FileName, Lattice, AtomID, ZBL]]
            # print("NewRow in RhoCalculator is:\n" + str(NewRow))
            # print("RhoNumpy in RhoCalculator Before adding a New One is:\n" + str(RhoNumpy))
            ZblConfigNumpy = np.append(ZblConfigNumpy, NewRow, axis=0)
            # print("RhoNumpy in RhoCalculator After adding a New One is:\n" + str(RhoNumpy))
        FileName = ZblConfigNumpy.mean(axis=0)[0]
        Lattice = ZblConfigNumpy.mean(axis=0)[1]
        ZblConfig = ZblConfigNumpy.sum(axis=0)[3]
        ZblResult = [[FileName, Lattice, ZblConfig]]


    return ZblResult


# </editor-fold>

# <editor-fold desc="RunLammps">
print("RunLammps")
def RunLammps(Num_V):
    LammpsTempFileAddress = LammpsTempAddress + "/" + LammpsTempFileName
    LammpsInFileName = str(Num_V) + ".in"
    LammpsInFileAddress = WorkingDirectory + "/" + LammpsInFileName

    shutil.copyfile(LammpsTempFileAddress, LammpsInFileAddress)
    # shutil.copyfile(TestDirectory + "/" + PotentialName + "_ASE.eampot", DimerDirectory + "/" + PotentialFileName)
    # shutil.copyfile(TestDirectory + "/" + "0_100000.restart", DimerDirectory + "/" + "0_100000.restart")

    with fileinput.FileInput(LammpsInFileAddress, inplace=True) as file:
        for line in file:
            line = line.replace("WorkingDirectoryTemp", WorkingDirectory)
            line = line.replace("PotentialTemp", PotentialFileName)
            line = line.replace("NumTemp", str(Num_V))
            line = line.replace("CurrentDirectoryTemp", CurrentDirectory)
            line = line.replace("DirectionDirectoryTemp", DirectionDirectory)
            line = line.replace("StageNameTemp", str(StageName_V))
            if MeasureRho_V:
                line = line.replace("RunLammpsEphTemp ", "")
            else:
                line = line.replace("RunLammpsEphTemp ", "#")
            if WriteLog_V:
                line = line.replace("WriteLogTemp ", "")
            else:
                line = line.replace("WriteLogTemp ", "#")
            if WriteDump_V:
                line = line.replace("WriteDumpTemp ", "")
            else:
                line = line.replace("WriteDumpTemp ", "#")
            if WriteReport_V:
                line = line.replace("WriteReportTemp ", "")
            else:
                line = line.replace("WriteReportTemp ", "#")
            print(line, end="")

    LammpsArgs = ["-screen", LammpsScreen]
    lmp = lammps(cmdargs=LammpsArgs)
    lmp.file(LammpsInFileAddress)
    return Num_V
# </editor-fold>

# <editor-fold desc="FunctionSelector">
def FunctionSelector(FunctionType_V, Direction_V, ReadEamEnergyFrom_V,
                     MeasureRho_V, DistRhoInterpolate_V,
                     RhoFInterpolate_V, DistPhiInterpolate_V,
                     SameDirectory_V,MirrorAtEq_V,
                     SkipHeader_V,DicRev_V):

    # <editor-fold desc="$$$$$ Type Dependency">
    if FunctionType_V == "Qsd":
        LookUp = Direction_V
    elif FunctionType_V == "Eos":
        LookUp = FunctionType_V

    # <editor-fold desc="Dictionary">

    DicInitialPosX = {
        "A": EdEosInitialPosXAlongA,
        "C": EdEosInitialPosXAlongC,
        "Ac": EdEosInitialPosXAlongAc,
    }

    DicBondLength = {
        "A": BondLengthAlongA,
        "C": BondLengthAlongC,
        "Ac": BondLengthAlongAc,
        "Eos": LatticeEquilibriumDist,
    }

    DicInitialPosY = {
        "A": EdEosInitialPosYAlongA,
        "C": EdEosInitialPosYAlongC,
        "Ac": EdEosInitialPosYAlongAc,
    }

    DicInitialPosZ = {
        "A": EdEosInitialPosZAlongA,
        "C": EdEosInitialPosZAlongC,
        "Ac": EdEosInitialPosZAlongAc,
    }

    DicRev = DicRev_V

    DicColumnDistance = {
        ".log": 1,
        ".dump": 14,
        ".report": 1,
    }

    DicColumnEnergy = {
        ".log": 7,
        ".dump": 19,
        ".report": 10,
    }

    DicColumnRho = {
        ".log": False,
        ".dump": 15,
        ".report": 11,
    }

    DicColumnFDer = {
        ".log": False,
        ".dump": False,
        ".report": 12,
    }

    DicEamMeasuringFunction = {
        ".log": LogExplorer,
        ".dump": EamCalculator,
        ".report": LammpsReportExplorer,
    }

    DicEamMeasuringFunctionHeader = {
        ".log": "FileName,Length,BoxX,BoxY,BoxZ,NumOfAtoms,EvEnergy,EnergyPerAtom,Rho",
        ".dump": "FileName,Lattice,BoxX1,BoxX2,BoxY1,BoxY2,BoxZ1,BoxZ2,id,type,x,y,z,c_eng,LatticeSizeNew,RhoCalculated,FCalculated,PhiCalculated,EnergyTotal,EnergyNewPerAtom",
        ".report": "Lattice,Step,Time,dt,Cpu,TemperatureIonic,Press,EnergyPotential,EnergyKinetic"
    }

    DicEamMeasuringFunctionArgs = {
        ".log": (FunctionType_V, Direction_V, DicRev[LookUp], MeasureRho_V, MirrorAtEq_V, SameDirectory_V),
        ".dump": (CutOff, DistRhoInterpolate_V, RhoFInterpolate_V, DistPhiInterpolate_V, DicBondLength[LookUp], False, 16, True, False, True),
        ".report": (FunctionType_V, DicRev[LookUp], DicBondLength[LookUp], "TimesCalled", TimesCalled, MirrorAtEq_V, MeasureRho_V, SkipHeader_V)
        # def LammpsReportExplorer(FileAddress_V, FunctionType_V , EdRev_V, BondLengthAlong_V, ReadIterationFrom_V, Iteration_V, MirrorAtEq_V, RunLammpsEph, EmbeddedFunction):
    }

    # </editor-fold>

    # <editor-fold desc="Selector">
    EamMeasuringFunction = DicEamMeasuringFunction[ReadEamEnergyFrom_V]
    EamMeasuringFunctionArgs = DicEamMeasuringFunctionArgs[ReadEamEnergyFrom_V]
    EamMeasuringFunctionHeader = DicEamMeasuringFunctionHeader[ReadEamEnergyFrom_V]
    EamMeasuringFunctionColumnDistance = DicColumnDistance[ReadEamEnergyFrom_V]
    EamMeasuringFunctionColumnEnergy = DicColumnEnergy[ReadEamEnergyFrom_V]
    EamMeasuringFunctionColumnRho = DicColumnRho[ReadEamEnergyFrom_V]
    EamMeasuringFunctionColumnFDer = DicColumnFDer[ReadEamEnergyFrom_V]

    # </editor-fold>

    return EamMeasuringFunction, EamMeasuringFunctionArgs, EamMeasuringFunctionHeader, EamMeasuringFunctionColumnDistance, EamMeasuringFunctionColumnEnergy, EamMeasuringFunctionColumnRho, EamMeasuringFunctionColumnFDer


# </editor-fold>

# <editor-fold desc="TempToInput">
def TempToInput(LammpsInFileAddress_V,WorkingDirectory_V,PotentialFileAddress_V,CurrentDirectory_V,DirectionDirectory_V,
                StageName_V,TimesCalled_V,RhoFinderName_V,
                InternalLoop_V,RunLammpsStart_V,RunLammpsFinish_V,
                RunLammpsEph_V,WriteLog_V,WriteDump_V,
                WriteReport_V,MeasureRho_V, MeasureDFer_V):
    with fileinput.FileInput(LammpsInFileAddress_V, inplace=True) as file:
        for line in file:
            line = line.replace("WorkingDirectoryTemp", WorkingDirectory_V)
            line = line.replace("PotentialFileAddressTemp", PotentialFileAddress_V)
            line = line.replace("NumTemp", str(TimesCalled_V))
            line = line.replace("CurrentDirectoryTemp", CurrentDirectory_V)
            line = line.replace("DirectionDirectoryTemp", DirectionDirectory_V)
            line = line.replace("StageNameTemp", str(StageName_V))
            line = line.replace("RhoFinderNameTemp", str(RhoFinderName_V))
            line = line.replace("RunLammpsStartTemp", str(RunLammpsStart_V))
            line = line.replace("RunLammpsFinishTemp", str(RunLammpsFinish_V))
            if InternalLoop_V:
                line = line.replace("InternalLoopTemp ", "")
                line = line.replace("ExternalLoopTemp ", "#")
            else:
                line = line.replace("InternalLoopTemp ", "#")
                line = line.replace("ExternalLoopTemp ", "")
            if MeasureRho_V:
                line = line.replace("MeasureRhoTemp ", "")
            else:
                line = line.replace("RunLammpsEphTemp ", "#")
            if RunLammpsEph_V:
                line = line.replace("RunLammpsEphTemp ", "")
            else:
                line = line.replace("RunLammpsEphTemp ", "#")
            if MeasureDFer_V:
                line = line.replace("MeasureDFerTemp ", "")
            else:
                line = line.replace("MeasureDFerTemp ", "#")
            if WriteLog_V:
                line = line.replace("WriteLogTemp ", "")
            else:
                line = line.replace("WriteLogTemp ", "#")
            if WriteDump_V:
                line = line.replace("WriteDumpTemp ", "")
            else:
                line = line.replace("WriteDumpTemp ", "#")

            if WriteReport_V and MeasureRho_V and MeasureDFer_V:
                line = line.replace("WriteReportRhoFDerTemp ", "")
                line = line.replace("WriteReportRhoTemp ", "#")
                line = line.replace("WriteReportEphTemp ", "#")
                line = line.replace("WriteReportTemp ", "#")
            elif WriteReport_V and MeasureRho_V:
                line = line.replace("WriteReportRhoFDerTemp ", "#")
                line = line.replace("WriteReportRhoTemp ", "")
                line = line.replace("WriteReportEphTemp ", "#")
                line = line.replace("WriteReportTemp ", "#")
            elif WriteReport_V and RunLammpsEph_V:
                line = line.replace("WriteReportRhoFDerTemp ", "#")
                line = line.replace("WriteReportRhoTemp ", "#")
                line = line.replace("WriteReportEphTemp ", "")
                line = line.replace("WriteReportTemp ", "#")
            elif WriteReport_V:
                line = line.replace("WriteReportRhoFDerTemp ", "#")
                line = line.replace("WriteReportRhoTemp ", "#")
                line = line.replace("WriteReportEphTemp ", "#")
                line = line.replace("WriteReportTemp ", "")
            print(line, end="")
# </editor-fold>

# <editor-fold desc="PotHeaderChanger">
def PotHeaderChanger(Address_V,
                     Line1_V = "",
                     PotentialName_V=PotentialName, StageName_V = StageName,
                     Date_V=None):
    # Open the file in read mode

    if Date_V is None or Date_V == "":
        Date = datetime.datetime.now()
    else:
        Date = Date_V

    with open(Address_V, 'r') as file:
        lines = file.readlines()

    # Modify the first three lines with new content
    if Line1_V == "":
        Line1 = "EAM potential file reparameterized by Amir Ghorbani (amir.ghorbani@queensu.ca) and Laurent Beland  based on:"
    else:
        Line1 = Line1_V
    lines[0] = Line1 + '\n'
    lines[1] = PotentialName_V + " at Stage: " + StageName_V + '\n'
    lines[2] = "Created in: " + str(Date) + '\n'

    # Write the modified content back to the file
    with open(Address_V, 'w') as file:
        file.writelines(lines)
# </editor-fold>

# <editor-fold desc="Cost Function">
print("Cost Function")

# CorePool = mp.Pool(CoreNumber)
# print(CorePool)
# os.system("pause")

def CostFunction(FittingNumpyYvalue_V,
                 FunctionType_V, DicPotential_V,
                 StageName_V, StageNameBefore_V, DirectionList_V,
                 PotentialName_V,
                 FittingNumpyXvalue_V, TransitionPoints_V,
                 DicEamDf_V, DicEamDfInterpolate_V,DicEamDfInterpolateValue_V,DicEamCriticalRead_V,
                 DicDftDf_V,DicDftDfInterpolate_V,DicDftDfInterpolateValue_V,
                 DicRunLammpsRange_V,DicCostRange_V, DicRev_V,
                 EamSource_V,  # Extract,ExtractEph,RunExtract,Read
                 ReadEamEnergyFrom_V,
                 CurrentDirectory_V, LammpsTempAddress_V,
                 DeepAnalysis_V, PrintSectionTitle_V, TimeRecord_V, LogDeviation_V, MultiProcessor_V,
                 RunLammpsOnlyCostRange_V, SameDirectory_V, MirrorAtEq_V,
                 MeasureRho_V, RunLammpsEph_V, RhoFinderName_V,MeasureDFer_V,
                 WriteLog_V, WriteDump_V, WriteReport_V,
                 CostMeasurement_V
                 ):

    # os.system("pause")
    # PrintSectionTitle_V=True

    # <editor-fold desc="**********  Global History">
    if PrintSectionTitle_V: print("**********  Global History")
    if TimeRecord_V: start_time_L1 = time.time()
    if CostMeasurement_V:
        global TimesCalled
        TimesCalled += 1
        if PrintSectionTitle_V: print("\n\n________________________________________________________ Start")
        if PrintSectionTitle_V: print("********** Global History")
        if PrintSectionTitle_V: print("TimesCalled = " + str(TimesCalled))
        # print("TimesCalled: " + str(TimesCalled))
    else:
        TimesCalled = 0
        global DicEamDf
        global DicEamDfInterpolate
        global DicEamDfInterpolateValue
        global DicDftDf
        global DicDftDfInterpolate
        global DicDftDfInterpolateValues
        # global DicPotential
    # </editor-fold>

    # <editor-fold desc="**********  Properties">
    if PrintSectionTitle_V: print("**********  Properties")
    if TimeRecord_V: start_time_L1 = time.time()

    # <editor-fold desc="^^^^^ Cost Option">
    if PrintSectionTitle_V: print("^^^^^ Cost Option")
    if CostMeasurement_V:
        # PotentialFileAddress = PotentialName_V
        # print("TimesCalled: " + str(TimesCalled))
        FunctionFolderName = "Minimization"
    else:
        # PotentialFileAddress = PotentialName_V
        DistRhoInterpolate = ""
        RhoFInterpolate =""
        DistPhiInterpolate =""
        FunctionFolderName = "OnTheFly"
    # </editor-fold>

    # <editor-fold desc="^^^^^ Error Center">
    if PrintSectionTitle_V: print("^^^^^ Error Center")
    if ReadEamEnergyFrom_V == ".dump" and WriteDump_V == False:
        print("Inconsistancy in ReadEamEnergyFrom_V and WriteDump_V")
    # </editor-fold>

    # <editor-fold desc="^^^^^ Resetting">
    if PrintSectionTitle_V: print("^^^^ Resetting")
    ZeroArray = np.array([[0, 0], [1, 0]])
    QsdAlongALogNumpyInterpolateDistEnergy = PchipFunc(ZeroArray[:, 0], ZeroArray[:, 1], extrapolate=True)
    QsdAlongCLogNumpyInterpolateDistEnergy = PchipFunc(ZeroArray[:, 0], ZeroArray[:, 1], extrapolate=True)
    QsdAlongAcLogNumpyInterpolateDistEnergy = PchipFunc(ZeroArray[:, 0], ZeroArray[:, 1], extrapolate=True)
    EosLogNumpyInterpolateDistEnergy = PchipFunc(ZeroArray[:, 0], ZeroArray[:, 1], extrapolate=True)
    QsdAlongALogNumpyInterpolateRhoEnergy = PchipFunc(ZeroArray[:, 0], ZeroArray[:, 1], extrapolate=True)
    QsdAlongCLogNumpyInterpolateRhoEnergy = PchipFunc(ZeroArray[:, 0], ZeroArray[:, 1], extrapolate=True)
    QsdAlongAcLogNumpyInterpolateRhoEnergy = PchipFunc(ZeroArray[:, 0], ZeroArray[:, 1], extrapolate=True)
    EosLogNumpyInterpolateRhoEnergy = PchipFunc(ZeroArray[:, 0], ZeroArray[:, 1], extrapolate=True)
    QsdAlongALogNumpyInterpolateDistRho = PchipFunc(ZeroArray[:, 0], ZeroArray[:, 1], extrapolate=True)
    QsdAlongCLogNumpyInterpolateDistRho = PchipFunc(ZeroArray[:, 0], ZeroArray[:, 1], extrapolate=True)
    QsdAlongAcLogNumpyInterpolateDistRho = PchipFunc(ZeroArray[:, 0], ZeroArray[:, 1], extrapolate=True)
    EosLogNumpyInterpolateDistRho = PchipFunc(ZeroArray[:, 0], ZeroArray[:, 1], extrapolate=True)
    QsdAlongADftInterpolateRhoEnergy = PchipFunc(ZeroArray[:, 0], ZeroArray[:, 1], extrapolate=True)
    QsdAlongCDftInterpolateRhoEnergy = PchipFunc(ZeroArray[:, 0], ZeroArray[:, 1], extrapolate=True)
    QsdAlongAcDftInterpolateRhoEnergy = PchipFunc(ZeroArray[:, 0], ZeroArray[:, 1], extrapolate=True)
    EosDftInterpolateRhoEnergy= PchipFunc(ZeroArray[:, 0], ZeroArray[:, 1], extrapolate=True)
    DftInterpolate = PchipFunc(ZeroArray[:, 0], ZeroArray[:, 1], extrapolate=True)
    # </editor-fold>

    # <editor-fold desc="^^^^^ Dictionary">
    if PrintSectionTitle_V: print("^^^^^ Dictionary")

    # <editor-fold desc="FunctionTypeList">
    TypeList = DicTypeList[FunctionType_V]
    # </editor-fold>

    # <editor-fold desc="Eam">
    # print("Eam")
    DicEamDfStage = DicEamDf_V

    DicEamDfInterpolateStage = DicEamDfInterpolate_V

    DicEamDfInterpolateValueStage = DicEamDfInterpolateValue_V
    # </editor-fold>

    # <editor-fold desc="Critical">
    # print("Critical")
    if not CostMeasurement_V:
        DicEamCriticalRead = DicEamCriticalRead_V
#         DicEamCriticalStage = {
#             StageName: {
#         "A": {"Step": {"Min":"","Max":"","Eq":""}, "Dist": {"Min":"","Max":"","Eq":""}, "Rho":{"Min":"","Max":"","Eq":""}, "Energy":{"Min":"","Max":"","Eq":""}},
#         "C": {"Step": {"Min":"","Max":"","Eq":""}, "Dist": {"Min":"","Max":"","Eq":""}, "Rho":{"Min":"","Max":"","Eq":""}, "Energy":{"Min":"","Max":"","Eq":""}},
#         "Ac": {"Step": {"Min":"","Max":"","Eq":""}, "Dist": {"Min":"","Max":"","Eq":""}, "Rho":{"Min":"","Max":"","Eq":""}, "Energy":{"Min":"","Max":"","Eq":""}},
#         "Eos": {"Step": {"Min":"","Max":"","Eq":""}, "Dist": {"Min":"","Max":"","Eq":""}, "Rho":{"Min":"","Max":"","Eq":""}, "Energy":{"Min":"","Max":"","Eq":""}}
#     }
# }
    # </editor-fold>

    # <editor-fold desc="Dft">
    # print("Dft")

    DicDftDf = DicDftDf_V
    DicDftDfStage = {
        StageName: {
            "A": {"Dist": "", "Rho": "", "DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "",
                  "RhoEnergy": ""},
            "C": {"Dist": "", "Rho": "", "DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "",
                  "RhoEnergy": ""},
            "Ac": {"Dist": "", "Rho": "", "DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "",
                   "RhoEnergy": ""},
            "Eos": {"Dist": "", "Rho": "", "DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "",
                    "RhoEnergy": ""},
        }
    }

    DicDftDfInterpolate = DicDftDfInterpolate_V
    DicDftDfInterpolateStage = {
        StageName: {
            "A": {"Dist": "", "Rho": "", "DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "",
                  "RhoEnergy": ""},
            "C": {"Dist": "", "Rho": "", "DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "",
                  "RhoEnergy": ""},
            "Ac": {"Dist": "", "Rho": "", "DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "",
                   "RhoEnergy": ""},
            "Eos": {"Dist": "", "Rho": "", "DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "",
                    "RhoEnergy": ""},
        }
    }

    DicDftDfInterpolateValues = DicDftDfInterpolateValue_V
    DicDftDfInterpolateValuesStage = {
        StageName: {
            "A": {"Dist": "", "Rho": "", "DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "",
                  "RhoEnergy": ""},
            "C": {"Dist": "", "Rho": "", "DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "",
                  "RhoEnergy": ""},
            "Ac": {"Dist": "", "Rho": "", "DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "",
                   "RhoEnergy": ""},
            "Eos": {"Dist": "", "Rho": "", "DistRho": "", "DistF": "", "DistFDer": "", "DistPhi": "", "DistEnergy": "",
                    "RhoEnergy": ""},
        }
    }
    # </editor-fold>

    # <editor-fold desc="Cost">
    # print("Cost")

    DicRunLammpsRange = DicRunLammpsRange_V

    DicCostRange = DicCostRange_V

    DicRev = DicRev_V

    # </editor-fold>

    # <editor-fold desc="Potential">
    # print("Potential")
    if not CostMeasurement_V:
        DicPotential = DicPotential_V

        DicPotentialStage = {
            StageName: {
                "PhysicalDist": "", "RhoDist": "", "Rho": "", "F": "", "Phi": "",
                "DistRhoInterpolate": "", "RhoFInterpolate": "", "DistPhiInterpolate": "",
                "DistRhoInterpolateValues": "", "RhoFInterpolateValues": "", "DistPhiInterpolateValues": ""
            }
        }

    # </editor-fold>
    if TimeRecord_V: print("________________________________________________________" + str(
        time.time() - start_time_L1) + "seconds"); start_time_L1 = time.time()
    # </editor-fold>
    # </editor-fold>

    # <editor-fold desc="**********  Folder Making">
    if PrintSectionTitle_V: print("**********  Folder Making")
    if TimeRecord_V: start_time_L1 = time.time()
    CostFunctionDirectory = CurrentDirectory_V + "/" + FunctionType_V + "/" + StageName_V + "/" + FunctionFolderName + "/"
    Path(CostFunctionDirectory).mkdir(parents=True, exist_ok=True)
    if TimeRecord_V: print("________________________________________________________" + str(
        time.time() - start_time_L1) + "seconds"); start_time_L1 = time.time()
    # </editor-fold>

    # <editor-fold desc="**********  Basis: Dist or Rho">
    if PrintSectionTitle_V: print("**********  Basis: Dist or Rho")
    if (MeasureRho_V) or (RunLammpsEph_V):
        Variable = "Rho"
        VariableEnergy = "RhoEnergy"
        VariableDist = "RhoDist"
        VariableAxis = "Rho"
        VariableGridName = "RhoDist"
        SkipHeader = 2
        # <editor-fold desc="Function Dependency">
        if FunctionType_V == "Qsd":
            FunctionName = "Phi"
            FunctionInterpolateName = "RhoPhiInterpolate"
            FunctionInterpolateValuesName = "RhoPhiInterpolateValues"
        elif FunctionType_V == "Eos":
            FunctionName = "F"
            FunctionInterpolateName = "RhoFInterpolate"
            FunctionInterpolateValuesName = "RhoFInterpolateValues"
            # </editor-fold>
    else:
        Variable = "Dist"
        VariableEnergy = "DistEnergy"
        VariableDist = "PhysicalDist"
        VariableAxis = r"R " + "(" + "Å" + r")"
        VariableGridName = "PhysicalDist"
        SkipHeader = 1
        # <editor-fold desc="Function Dependency">
        if FunctionType_V == "Qsd":
            FunctionName = "Phi"
            FunctionInterpolateName = "DistPhiInterpolate"
            FunctionInterpolateValuesName = "DistPhiInterpolateValues"

        elif FunctionType_V == "Eos":
            FunctionName = "F"
            FunctionInterpolateName = "DistFInterpolate"
            FunctionInterpolateValuesName = "DistFInterpolateValues"
        # </editor-fold>
    # </editor-fold>

    # <editor-fold desc="**********  Potential">
    if PrintSectionTitle_V: print("**********  Potential")
    if TimeRecord_V: start_time_L1 = time.time()

    # <editor-fold desc="^^^^^ DicRead">
    if PrintSectionTitle_V: print("^^^^^ DicRead")
    if not CostMeasurement_V:
        if StageName != "Original":
            PotentialVariableFunctionBefore = DicPotential[StageNameBefore_V][FunctionName]
            PotentialVariableGridBefore = DicPotential[StageNameBefore_V][VariableGridName]
            PotentialVariableFunctionInterpolateBefore = DicPotential[StageNameBefore_V][FunctionInterpolateName]
            PotentialVariableFunctionInterpolateValuesBefore = DicPotential[StageNameBefore_V][FunctionInterpolateValuesName]

    # </editor-fold>

    # <editor-fold desc="^^^^^ Reading Potential - Initial">
    if PrintSectionTitle_V: print("^^^^^ Reading Potential - Initial")

    # <editor-fold desc="Ase">
    if PrintSectionTitle_V: print("CostFunction-Ase")

    PotentialFileAddress = PotentialName_V

    # print(PotentialFileAddress)
    # os.system("pause")

    Potential = EAM(potential=PotentialFileAddress, form="alloy")

    ElementNo = 0

    # In potential file, the 5th line is: Nrho, drho, Nr, dr, Cutoff
    Nrho = Potential.nrho  # 5000 # number of tabulated values for embedding function F(rho)
    # print("Nrho is : " + str(Nrho))
    drho = Potential.drho  # 1.0000000000000000E-003 # spacing in density
    # print("drho is : " + str(drho))
    Nr = Potential.nr  # 5000 # number of tabulated values for effective charge function Z(r)
    # print("Nr is : " + str(Nr))
    dr = Potential.dr  # 1.5199999999999999E-003 # distance space for pair interaction and density in Angstrom
    # print("dr is : " + str(dr))
    LatticeParameter = Potential.a[ElementNo]  # distance space for pair interaction and density in Angstrom
    # print("LatticeParameter is : " + str(LatticeParameter))

    Cutoff = round(dr * Nr, 6)  # 7.6E+000
    # print(Cutoff)

    LastRho = Nrho * drho
    LastDist = Nr * dr
    PhysicalDist = np.mgrid[0:LastDist:dr]
    RhoDist = np.mgrid[0:LastRho:drho]

    # Electron Density
    PotentialRho = Potential.electron_density[ElementNo](PhysicalDist)
    # print("OriginalPotentialRho is:")
    # print(OriginalPotentialRho)
    # print("Shape of OriginalPotentialRho is: " + str(np.shape(OriginalPotentialRho)))

    # Embedded Energy
    PotentialF = Potential.embedded_energy[ElementNo](RhoDist)
    # print("PotentialF is:")
    # print(PotentialF)
    # print("Shape of PotentialF is: " + str(np.shape(PotentialF)))

    # Pair Interaction
    PotentialPhi = Potential.phi[ElementNo, ElementNo](PhysicalDist)
    # print("PotentialPhi is:")
    # print(PotentialPhi)
    # print("Shape of PotentialPhi is: " + str(np.shape(OriginalPotentialRho)))

    # </editor-fold>

    # <editor-fold desc="Interpolation-Spline">
    DistRhoInterpolate = CubicSpline(PhysicalDist, PotentialRho)
    RhoFInterpolate = CubicSpline(RhoDist, PotentialF)
    DistPhiInterpolate = CubicSpline(PhysicalDist, PotentialPhi)

    # add_boundary_knots(DistRhoInterpolate, 1, 2)
    # add_boundary_knots(RhoFInterpolate, 1, 2)
    # add_boundary_knots(DistPhiInterpolate, 1, 2)

    if not CostMeasurement_V:
        DistRhoInterpolateValues = DistRhoInterpolate(PhysicalDist)
        RhoFInterpolateValues = RhoFInterpolate(RhoDist)
        DistPhiInterpolateValues = DistPhiInterpolate(PhysicalDist)
    # </editor-fold>

    # <editor-fold desc="$$$$$ Type Dependency">
    if FunctionType_V == "Qsd":
        Function = DistPhiInterpolate
        XNumpy = PhysicalDist
        YNumpy = PotentialPhi
        Dx = dr
        DD = dr
        if not CostMeasurement_V:
            FunctionValues = DistPhiInterpolateValues
    elif FunctionType_V == "Eos":
        Function = RhoFInterpolate
        XNumpy = RhoDist
        YNumpy = PotentialF
        Dx = drho
        DD = drho
        if not CostMeasurement_V:
            FunctionValues = RhoFInterpolateValues
    # </editor-fold>

    Plotting = False
    PlottingShow = False
    # <editor-fold desc="Plotting">
    if PrintSectionTitle_V: print("CostFunction-Plotting")
    if Plotting:
        Title = StageName_V + "-" + FunctionName + "-" + str(TimesCalled)
        plt.scatter(XNumpy, YNumpy, color=Colors[0], label="YNumpy")
        plt.plot(XNumpy,Function(XNumpy), color=Colors[1], label="Function")
        plt.xlabel("XValue")
        plt.ylabel("YValue")
        plt.title(Title)
        # plt.ylim(Function(TransitionMax) - 100, Function(TransitionMin) + 100)
        # plt.xlim(1, 150)
        # plt.ylim(Function(TransitionMax) - 100, Function(TransitionMin) + 100)
        # plt.xscale("log")
        # plt.yscale("log")
        # plt.grid()
        plt.legend()
        plt.savefig(CostFunctionDirectory + "/" + Date + "-CostFunc-" + Title)
        if PlottingShow:
            plt.show()
        else:
            plt.close()
    # </editor-fold>
    Plotting = False
    PlottingShow = False

    # </editor-fold>

    # <editor-fold desc="^^^^^ Modification">
    if PrintSectionTitle_V: print("^^^^^ Modification")
    if CostMeasurement_V:
        # <editor-fold desc="^^^^^ Refitting">
        if PrintSectionTitle_V: print("^^^^^ Refitting")

        # <editor-fold desc="Fitting Points">
        if PrintSectionTitle_V: print("CostFunction-Fitting Points")
        MinCounter = 0
        # print(FittingDistNumpy)
        # os.system("pause")
        FittingNumpyXvalueMin = np.min(FittingNumpyXvalue_V)
        FittingNumpyXvalueMax = np.max(FittingNumpyXvalue_V)
        FittingNumpyYvalueMin = np.min(FittingNumpyYvalue_V)
        FittingNumpyYvalueMax = np.max(FittingNumpyYvalue_V)

        FittingNumpyXvalueCount = FittingNumpyXvalue_V.shape[0]
        # print(FittingNumpyXvalueCount)
        FittingDistLen = FittingNumpyXvalueMax - FittingNumpyXvalueMin
        FittingDistDr = FittingDistLen / FittingNumpyXvalueCount

        FittingNumpy = np.stack((FittingNumpyXvalue_V, FittingNumpyYvalue_V), axis=1)

        # FittingNumpyTimesCalled = np.hstack((FittingNumpy, np.ones((FittingNumpy.shape[0], 1))))
        # FittingNumpyTimesCalled[:, 2] = FittingNumpyTimesCalled[:, 2] * TimesCalled
        FittingNumpyTimesCalled = np.insert(FittingNumpy, 2, values=TimesCalled, axis=1)

        if DeepAnalysis_V:
            FittingNumpyTimesCalledDf = pd.DataFrame(data=FittingNumpyTimesCalled,
                                                     columns=["Xvalue", "Yvalue", "TimesCalled"])
            # print(FittingNumpyTimesCalledDf)
            FittingNumpyTimesCalledDf.to_csv(
                CostFunctionDirectory + "/" + Date + "-CostFunc-" + StageName_V + "-FittingNumpyTimesCalled.csv",
                mode="a", index=False, header=False)

        # print(FittingNumpyTimesCalledDf)
        # </editor-fold>

        # <editor-fold desc="Transition">
        if PrintSectionTitle_V: print("CostFunction-Transition")

        if isinstance(TransitionPoints_V, str):
            FittingDistDrMin = FittingNumpyXvalue_V[1] - FittingNumpyXvalue_V[0]
            FittingDistDrMax = FittingNumpyXvalue_V[-1] - FittingNumpyXvalue_V[-2]
        else:
            TransitionMin = TransitionPoints_V[0]
            TransitionMax = TransitionPoints_V[1]
        UnchangedNumpy = np.zeros((0, 2))
        for Row in range(len(XNumpy)):
            X = XNumpy[Row]
            if X == 0:
                continue
            elif X < TransitionMin:
                Y = Function(X)
                FunctionValue = np.array([[X, Y]])
                # print(FunctionValue)
                UnchangedNumpy = np.append(UnchangedNumpy, FunctionValue, axis=0)
            elif TransitionMin <= X < TransitionMax:
                continue
            elif TransitionMax < X:
                Y = Function(X)
                # print(Y)
                FunctionValue = np.array([[X, Y]])
                UnchangedNumpy = np.append(UnchangedNumpy, FunctionValue, axis=0)
        # print(PhiMinNumpyOuterRegions)
        FunctionNumpy = UnchangedNumpy
        # print(FunctionNumpy)
        # </editor-fold>

        # <editor-fold desc="Stacking">
        if PrintSectionTitle_V: print("CostFunction-Stacking")
        FunctionNumpy = np.concatenate((FunctionNumpy, FittingNumpy), axis=0)
        # </editor-fold>

        # <editor-fold desc="Filtering">
        if PrintSectionTitle_V: print("Rounding")
        FunctionNumpy[np.absolute(FunctionNumpy[:, 1]) < 1e-5, 1] = 0
        # </editor-fold>

        # <editor-fold desc="Exporting">
        if PrintSectionTitle_V: print("CostFunction-Exporting")
        if DeepAnalysis_V:
            np.savetxt(CostFunctionDirectory + "/" + Date + "-CostFunc-" + StageName_V + "-FunctionNumpy-" + str(
                TimesCalled) + ".csv", FunctionNumpy,
                       delimiter=",", header="Xvalue,Yvalue", comments="")
        # </editor-fold>

        # <editor-fold desc="Interpolation-PchipValue">
        if PrintSectionTitle_V: print("CostFunction-Interpolation-PchipValue")
        FunctionNumpy = FunctionNumpy[FunctionNumpy[:, 0].argsort()]
        # print(FunctionNumpy)
        FunctionNumpyInterpolate = PchipFunc(FunctionNumpy[:, 0], FunctionNumpy[:, 1], extrapolate=True)
        # add_boundary_knots(FunctionNumpyInterpolate, 1, 2)
        # FunctionNumpyInterpolateValue = PchipValue(FunctionNumpy[:, 0], FunctionNumpy[:, 1], XNumpy)
        FunctionNumpyInterpolateValue = FunctionNumpyInterpolate(XNumpy)
        # </editor-fold>

        # # <editor-fold desc="Exporting">
        # if PrintSectionTitle_V: print("CostFunction-Exporting")
        # np.savetxt(Date + "-CostFunc-" + StageName_V + "-FunctionNumpyInterpolateValue-" + str(TimesCalled) + ".csv", FunctionNumpyInterpolateValue,
        #            delimiter=",", header="Yvalue", comments="")
        # # </editor-fold>
        if StageNamePure == "EgtSatZblPpmEpm":
            Plotting = False
            PlottingShow = False
        # <editor-fold desc="Plotting">
        if PrintSectionTitle_V: print("CostFunction-Plotting")
        if Plotting:
            Title = StageName_V + "-" + FunctionName + "-" + str(TimesCalled)
            plt.scatter(DicPotential_V[StageNameBefore_V][VariableDist], DicPotential_V[StageNameBefore_V][FunctionName],color=Colors[6], label="FuncBeforeInterpolate")
            plt.scatter(FunctionNumpy[:, 0], FunctionNumpy[:, 1], color=Colors[0], label="FunctionNumpy")
            plt.plot(XNumpy, FunctionNumpyInterpolateValue, color=Colors[0], label="FunctionNumpyInterpolateValue")
            plt.scatter(FittingNumpy[:, 0], FittingNumpy[:, 1], color=Colors[1], label="FittingNumpy")
            # plt.plot(XNumpy, FuncBeforeInterpolate_V(XNumpy), color=Colors[2], label="FuncBeforeInterpolate")
            plt.xlabel(VariableAxis)
            plt.ylabel(FunctionName)
            plt.title(Title)
            plt.xlim(DicVariable[FunctionType_V][Variable]["FunctionAxisXMin"], DicVariable[FunctionType_V][Variable]["FunctionAxisXMax"])
            plt.ylim(DicVariable[FunctionType_V][Variable]["FunctionAxisYMin"], DicVariable[FunctionType_V][Variable]["FunctionAxisYMax"])
            # plt.xscale("log")
            # plt.yscale("log")
            # plt.grid()
            plt.legend()
            plt.savefig(CostFunctionDirectory + "/" + Date + "-CostFunc-" + Title)
            if PlottingShow:
                plt.show()
            else:
                plt.close()
        # </editor-fold>
        Plotting = False
        PlottingShow = False
        # </editor-fold>

        # <editor-fold desc="^^^^^ Creating Potential">
        if PrintSectionTitle_V: print("^^^^^ Creating Potential")

        # <editor-fold desc="ASE">
        if PrintSectionTitle_V: print("CostFunction-ASE")

        ZrCrystal = bulk("Zr", "hcp", a=LatticeEquilibriumDist)

        # <editor-fold desc="$$$$$ Type Dependency">
        if FunctionType_V == "Qsd":
            Potential = EAM(
                elements=["Zr"], embedded_energy=np.array([RhoFInterpolate]),
                electron_density=np.array([DistRhoInterpolate]),
                phi=np.array([[FunctionNumpyInterpolate]]), cutoff=CutOff, form="alloy",
                # the following terms are only required to write out a file
                Z=[40], nr=Nr, nrho=Nrho, dr=dr, drho=drho,
                lattice=["hcp"], mass=[91.224], a=[LatticeEquilibriumDist])
        elif FunctionType_V == "Eos":
            Potential = EAM(
                elements=["Zr"], embedded_energy=np.array([FunctionNumpyInterpolate]),
                electron_density=np.array([DistRhoInterpolate]),
                phi=np.array([[DistPhiInterpolate]]), cutoff=CutOff, form="alloy",
                # the following terms are only required to write out a file
                Z=[40], nr=Nr, nrho=Nrho, dr=dr, drho=drho,
                lattice=["hcp"], mass=[91.224], a=[LatticeEquilibriumDist])
        # </editor-fold>

        ZrCrystal.calc = Potential
        PotentialAseEnergy = ZrCrystal.get_potential_energy()
        Potential.write_potential(PotentialFileAddress)
        PotHeaderChanger(PotentialFileAddress,StageName_V = StageName_V)

        # </editor-fold>

        # <editor-fold desc="Checking">
        if DeepAnalysis_V:
            PotentialCheckAse = EAM(potential=PotentialFileAddress, form="alloy")
            ZrCrystal.calc = PotentialCheckAse
            PotentialCheckAseEnergy = ZrCrystal.get_potential_energy()
            # print("Cohesive Energy for Zr = ", EgtSatZblAseEnergy, " eV")
            error = (PotentialAseEnergy - PotentialCheckAseEnergy) / PotentialAseEnergy
            # print("read/write check error = ", error)
            # if abs(error) < 1e-4:
            # print("Less")

        # </editor-fold>

        # <editor-fold desc="Plotting Potential">
        if PrintSectionTitle_V: print("CostFunction-Plotting Potential")
        if Plotting:
            PotentialCheckAse.plot()
            if PlottingShow:
                plt.show()
            else:
                plt.close()

        # </editor-fold>
        # </editor-fold>

        # <editor-fold desc="^^^^^ Reading Potential - Fitted (Necessary for read from Dump or Rho calculation)">
        if PrintSectionTitle_V: print("^^^^^ Reading Potential - Fitted (Necessary for read from Dump or Rho calculation)")
        if (ReadEamEnergyFrom_V == ".dump") or (MeasureRho_V) or (RunLammpsEph_V):

            # <editor-fold desc="Ase">
            if PrintSectionTitle_V: print("CostFunction-Ase")

            Potential = EAM(potential=PotentialFileAddress, form="alloy")

            ElementNo = 0

            # In potential file, the 5th line is: Nrho, drho, Nr, dr, Cutoff
            Nrho = Potential.nrho  # 5000 # number of tabulated values for embedding function F(rho)
            # print("Nrho is : " + str(Nrho))
            drho = Potential.drho  # 1.0000000000000000E-003 # spacing in density
            # print("drho is : " + str(drho))
            Nr = Potential.nr  # 5000 # number of tabulated values for effective charge function Z(r)
            # print("Nr is : " + str(Nr))
            dr = Potential.dr  # 1.5199999999999999E-003 # distance space for pair interaction and density in Angstrom
            # print("dr is : " + str(dr))
            LatticeParameter = Potential.a[
                ElementNo]  # 1.5199999999999999E-003 # distance space for pair interaction and density in Angstrom
            # print("LatticeParameter is : " + str(LatticeParameter))

            Cutoff = round(dr * Nr, 6)  # 7.6E+000
            # print(Cutoff)

            LastRho = Nrho * drho
            LastDist = Nr * dr
            PhysicalDist = np.mgrid[0:LastDist:dr]
            RhoDist = np.mgrid[0:LastRho:drho]

            # Electron Density
            PotentialRho = Potential.electron_density[ElementNo](PhysicalDist)
            # print("OriginalPotentialRho is:")
            # print(OriginalPotentialRho)
            # print("Shape of OriginalPotentialRho is: " + str(np.shape(OriginalPotentialRho)))

            # Embedded Energy
            PotentialF = Potential.embedded_energy[ElementNo](RhoDist)
            # print("PotentialF is:")
            # print(PotentialF)
            # print("Shape of PotentialF is: " + str(np.shape(PotentialF)))

            # Pair Interaction
            PotentialPhi = Potential.phi[ElementNo, ElementNo](PhysicalDist)
            # print("PotentialPhi is:")
            # print(PotentialPhi)
            # print("Shape of PotentialPhi is: " + str(np.shape(OriginalPotentialRho)))

            # EamMin = np.vstack((PhysicalDist, RhoDist, PotentialRho, PotentialF, PotentialPhi))
            # </editor-fold>

            # <editor-fold desc="Plotting Potential">
            if PrintSectionTitle_V: print("CostFunction-Plotting Potential")
            if Plotting:
                Potential.plot()
                if PlottingShow:
                    plt.show()

            # </editor-fold>

            # <editor-fold desc="Finding the Equilibrium Tally">
            FMin = np.min(PotentialF)
            FMinIndex = np.where(PotentialF == FMin)[0][0]
            # print(FOriginalMinIndex)
            # </editor-fold>

            # <editor-fold desc="Interpolation-Spline">
            DistRhoInterpolate = CubicSpline(PhysicalDist, PotentialRho)
            RhoFInterpolate = CubicSpline(RhoDist, PotentialF)
            DistPhiInterpolate = CubicSpline(PhysicalDist, PotentialPhi)

            # add_boundary_knots(DistRhoInterpolate, 1, 2)
            # add_boundary_knots(RhoFInterpolate, 1, 2)
            # add_boundary_knots(DistPhiInterpolate, 1, 2)

            DistRhoInterpolateValues = DistRhoInterpolate(PhysicalDist)
            RhoFInterpolateValues = RhoFInterpolate(RhoDist)
            DistPhiInterpolateValues = DistPhiInterpolate(PhysicalDist)
            # </editor-fold>

            # <editor-fold desc="$$$$$ Type Dependency">
            if FunctionType_V == "Qsd":
                Function = DistPhiInterpolate
                FunctionValues = DistPotentialPhiInterpolateValues
                XNumpy = PhysicalDist
                YNumpy = PotentialPhi
                Dx = dr
                DD = dr
            elif FunctionType_V == "Eos":
                Function = RhoFInterpolate
                FunctionValues = RhoFInterpolateValues
                XNumpy = RhoDist
                YNumpy = PotentialF
                Dx = drho
                DD = drho
            # </editor-fold>

            Plotting = False
            PlottingShow = False

            # <editor-fold desc="Plotting">
            if Plotting:
                Title = "DistRhoInterpolateValues"
                plt.plot(PhysicalDist, DistRhoInterpolateValues)
                plt.scatter(PhysicalDist, PotentialRho, color="r")
                plt.xlabel("Distance")
                plt.ylabel("Rho")
                plt.title(Title)
                # plt.xlim(0, 8)
                # plt.ylim(-10, 500)
                # plt.yscale("log")
                # plt.grid()
                plt.savefig(CostFunctionDirectory + "/" + Date + "-CostFunc-" + StageName_V + "-" + Title)
                if PlottingShow:
                    plt.show()

                Title = "RhoFInterpolateValues"
                plt.plot(RhoDist, RhoFInterpolateValues)
                plt.scatter(RhoDist, PotentialF, color="r")
                plt.xlabel("Rho")
                plt.ylabel("F")
                plt.title(Title)
                # plt.xlim(0, 8)
                # plt.ylim(-10, 500)
                # plt.yscale("log")
                plt.savefig(CostFunctionDirectory + "/" + Date + "-CostFunc-" + StageName_V + "-" + Title)
                if PlottingShow:
                    plt.show()

                Title = "DistPotentialPhiInterpolateValues"
                plt.plot(PhysicalDist, DistPotentialPhiInterpolateValues)
                plt.scatter(PhysicalDist, PotentialPhi, color="r")
                plt.xlabel("Distance")
                plt.ylabel("V")
                plt.title(Title)
                # plt.xlim(0, 8)
                # plt.ylim(-10, 500)
                # plt.yscale("log")
                plt.savefig(CostFunctionDirectory + "/" + Date + "-CostFunc-" + StageName_V + "-" + Title)
                if PlottingShow:
                    plt.show()
            # </editor-fold>

            Plotting = False
            PlottingShow = False

            # <editor-fold desc="Exporting">
            if DeepAnalysis_V:
                PotentialNumpy = np.stack((PhysicalDist, PotentialRho, PotentialPhi), axis=-1)
                np.savetxt(
                    CostFunctionDirectory + "/" + Date + "-CostFunc-" + StageName_V + "-PotentialNumpyRhoPhi-" + str(
                        TimesCalled) + ".csv", PotentialNumpy,
                    delimiter=",", header="PhysicalDist,PotentialRho,PotentialPhi", comments="")
                PotentialNumpy = np.stack((RhoDist, PotentialF), axis=-1)
                np.savetxt(CostFunctionDirectory + "/" + Date + "-CostFunc-" + StageName_V + "-PotentialNumpyF-" + str(
                    TimesCalled) + ".csv", PotentialNumpy,
                           delimiter=",", header="RhoDist,PotentialF", comments="")
            # </editor-fold>

            # <editor-fold desc="Rho Finder">
            if PrintSectionTitle_V: print("CostFunction-Rho Finder")
            if RunLammpsEph_V:
                Nbeta = 10

                Line1 = "Rho Finder-Based on potential: " + PotentialName + "_" + StageName_V + ".eampot"
                Line2 = "Line2: Nrho (PointsForRho), dr, Nbeta (PointsForBeta), drho, Cutoff"
                Line3 = "Body consists: Nrho lines of Rho + Nbeta lines of beta"
                Line4 = "!1 Zr"
                Line5 = "!" + str(Nrho) + " " + str(dr) + " " + str(Nbeta) + " " + str(drho) + " " + str(CutOff)
                Line6 = "!" + str(40)
                RhoFinderHeader = str(Line1) + "\n" + \
                                  str(Line2) + "\n" + \
                                  str(Line3) + "\n" + \
                                  str(Line4) + "\n" + \
                                  str(Line5) + "\n" + \
                                  str(Line6)

                RhoFileName = StageName_V + "-RhoFinder.beta"
                # RhoFileAddress = CostFunctionDirectory + "/" + RhoFileName
                RhoFileAddress = CurrentDirectory_V + "/" + RhoFileName
                np.savetxt(RhoFileAddress, PotentialRho, newline="\n", header=RhoFinderHeader, fmt="%.8e")

                with fileinput.FileInput(RhoFileAddress, inplace=True) as file:
                    for line in file:
                        print(line.replace("# !", ""), end="")

                ZeroBetas = ""
                for row in range(Nbeta):
                    ZeroBetas += str(0) + "\n"

                with open(RhoFileAddress, "a") as MyFile:
                    MyFile.write(ZeroBetas)
            # </editor-fold>

        # </editor-fold>

        # </editor-fold>
    # </editor-fold>

    # <editor-fold desc="^^^^^ DicUpdate">
    if PrintSectionTitle_V: print("^^^^^ DicUpdate")
    if not CostMeasurement_V:
        DicPotentialStage[StageName_V]["RhoDist"] = RhoDist
        DicPotentialStage[StageName_V]["PhysicalDist"] = PhysicalDist
        DicPotentialStage[StageName_V]["Rho"] = PotentialRho
        DicPotentialStage[StageName_V]["F"] = PotentialF
        DicPotentialStage[StageName_V]["Phi"] = PotentialPhi
        DicPotentialStage[StageName_V]["DistRhoInterpolate"] = DistRhoInterpolate
        DicPotentialStage[StageName_V]["RhoFInterpolate"] = RhoFInterpolate
        DicPotentialStage[StageName_V]["DistPhiInterpolate"] = DistPhiInterpolate
        DicPotentialStage[StageName_V]["DistRhoInterpolateValues"] = DistRhoInterpolateValues
        DicPotentialStage[StageName_V]["RhoFInterpolateValues"] = RhoFInterpolateValues
        DicPotentialStage[StageName_V]["DistPhiInterpolateValues"] = DistPhiInterpolateValues

    # </editor-fold>

    if TimeRecord_V: print("________________________________________________________" + str(
        time.time() - start_time_L1) + "seconds"); start_time_L1 = time.time()
    # </editor-fold>

    # <editor-fold desc="**********  Running and Extraction">
    if PrintSectionTitle_V: print("**********  Running and Extraction")
    if TimeRecord_V: start_time_L1 = time.time()
    # <editor-fold desc="$$$$$ Type Dependency">
    if FunctionType_V == "Qsd":
        # <editor-fold desc="^^^^^ Qsd">
        if PrintSectionTitle_V: print("^^^^^ Qsd")

        if EamSource_V == "Extract":

            # <editor-fold desc="AlongA">
            if PrintSectionTitle_V: print("CostFunction-AlongA")

            InitialPosX = EdEosInitialPosXAlongA
            InitialPosY = EdEosInitialPosYAlongA
            InitialPosZ = EdEosInitialPosZAlongA

            CostFuncQsdAlongADumpNumpy = np.zeros((0, 20))
            CostFuncQsdAlongALogNumpy = np.zeros((0, 9))
            ErrorFree = True

            # print("EosMinNumpy is:\n" + str(EosMinNumpy))
            NameLatticeBoxConfigNumpyAll = np.zeros((0, 14))

            for root, dirs, files in os.walk(FolderAddressEamEdPpmAlongA, topdown=False):  # open the files
                # print("root is: " + str(root))
                # print("files are: " + str(files))
                # print("dirs is: " + str(dirs))
                for name in files:
                    # print("File Name is: " + str(name))
                    if ".log" in name:  # target the dump files
                        # print("Log File opened: " + str(name))
                        LogFileAddress = os.path.join(root, name)
                        # def LogExplorer(LogFileAddress_V, Type_V, Along_V, Rev_V , RhoFinder_V,MirrorAtEq_V,SameDirectory_V):
                        EamLogNew = LogExplorer(LogFileAddress, EmbeddedDimer_V=True, Along_V="AlongA", EOS_V=False)

                        # print(EamLogNew)
                        if EamLogNew is False:
                            ErrorFree = False
                        else:
                            # print("EamLogNew is: " + str(EamLogNew))
                            # LogResult is ([[FileName, Length, BoxX, BoxY, BoxZ, NumOfAtoms, EvEnergy, EnergyPerAtom]])
                            CostFuncQsdAlongALogNumpy = np.append(CostFuncQsdAlongALogNumpy, EamLogNew, axis=0)
                            ErrorFree = True

                    elif ".dump" in name:  # target the dump files
                        # print("Dump File opened: " + str(name))
                        # print("root is: " + str(root))
                        # print(root.split("\\")[-1])
                        FolderName = int(root.split("\\")[-1])
                        # print("Folder is: " + str(FolderName))
                        if FolderName in range(ExtractingDumpRangeQsdEamAlongAStart,
                                               ExtractingDumpRangeQsdEamAlongAFinish):
                            # print("DUMP FILE OPENED: " + str(name))
                            # print(FileNameInRange)
                            DumpFileAddress = os.path.join(root, name)
                            # print(DumpFileAddress)
                            # EAM input is: (DumpFileAddress_V, CutOff_V, BondLenght_V, DistPhiSpline_V, RhoFSpline_V, DistVSpline_V, EmbeddedDimer_V = False, EmbeddedAtomNumber_V = 16, EOS = False)
                            EamNew = EamCalculator(DumpFileAddress, CutOff,
                                                   DistRhoInterpolate, RhoFInterpolate,
                                                   DistPhiInterpolate,
                                                   BondLenght_V=BondLengthAlongA,
                                                   EmbeddedDimer_V=True, EmbeddedAtomNumber_V=EdEosMovingIndexAlongA)[0]
                            # EamDimerResult is: FileName,Lattice,BoxX1,BoxX2,BoxY1,BoxY2,BoxZ1,BoxZ2,id,type,x,y,z,c_eng,DimerDistance, RhoCalculated, FCalculated, PhiCalculated, EnergyTotal, EnergyNewPerAtom
                            # print("EamNew is:\n" + str(EamNew))
                            CostFuncQsdAlongADumpNumpy = np.append(CostFuncQsdAlongADumpNumpy, EamNew, axis=0)

            # print("EamAlongANumpy is:\n" + str(EamAlongANumpy))

            CostFuncQsdAlongALogNumpy[:, 7] = CostFuncQsdAlongALogNumpy[:, 7] - \
                                              np.min(CostFuncQsdAlongALogNumpy, axis=0)[7]
            CostFuncQsdAlongADumpNumpy[:, 19] = CostFuncQsdAlongADumpNumpy[:, 19] - \
                                                np.min(CostFuncQsdAlongADumpNumpy, axis=0)[19]
            # print("Min of EosMinNumpy is: " + str(np.min(EamNumpy,axis=0)[3]))
            # </editor-fold>

            # <editor-fold desc="AlongC">
            if PrintSectionTitle_V: print("CostFunction-AlongC")

            InitialPosX = EdEosInitialPosXAlongC
            InitialPosY = EdEosInitialPosYAlongC
            InitialPosZ = EdEosInitialPosZAlongC

            CostFuncQsdAlongCDumpNumpy = np.zeros((0, 20))
            CostFuncQsdAlongCLogNumpy = np.zeros((0, 9))
            ErrorFree = True

            # print("EosMinNumpy is:\n" + str(EosMinNumpy))
            NameLatticeBoxConfigNumpyAll = np.zeros((0, 14))

            for root, dirs, files in os.walk(FolderAddressEamEdPpmAlongC, topdown=False):  # open the files
                # print("root is: " + str(root))
                # print("files are: " + str(files))
                # print("dirs is: " + str(dirs))
                for name in files:
                    # print("File Name is: " + str(name))
                    if ".log" in name:  # target the dump files
                        # print("Log File opened: " + str(name))
                        LogFileAddress = os.path.join(root, name)
                        # LogExplorer(LogFileAddress_V, EmbeddedDimer_V = False, Along_V = "AlongC", EOS_V = False):
                        EamLogNew = LogExplorer(LogFileAddress, EmbeddedDimer_V=True, Along_V="AlongC", EOS_V=False)
                        # print(EamLogNew)
                        if EamLogNew is False:
                            ErrorFree = False
                        else:
                            # print("EamLogNew is: " + str(EamLogNew))
                            # LogResult is ([[FileName, Length, BoxX, BoxY, BoxZ, NumOfAtoms, EvEnergy, EnergyPerAtom]])
                            CostFuncQsdAlongCLogNumpy = np.append(CostFuncQsdAlongCLogNumpy, EamLogNew, axis=0)
                            ErrorFree = True

                    elif ".dump" in name:  # target the dump files
                        # print("Dump File opened: " + str(name))
                        # print("root is: " + str(root))
                        # print(root.split("\\")[-1])
                        FolderName = int(root.split("\\")[-1])
                        # print("Folder is: " + str(FolderName))
                        if FolderName in range(ExtractingDumpRangeQsdEamAlongCStart,
                                               ExtractingDumpRangeQsdEamAlongCFinish):
                            # print("DUMP FILE OPENED: " + str(name))
                            # print(FileNameInRange)
                            DumpFileAddress = os.path.join(root, name)
                            # print(DumpFileAddress)
                            # EAM input is: (DumpFileAddress_V, CutOff_V, BondLenght_V, DistPhiSpline_V, RhoFSpline_V, DistVSpline_V, EmbeddedDimer_V = False, EmbeddedAtomNumber_V = 16, EOS = False)
                            EamNew = EamCalculator(DumpFileAddress, CutOff, DistRhoInterpolate,
                                                   RhoFInterpolate, DistPhiInterpolate,
                                                   BondLenght_V=BondLengthAlongC,
                                                   EmbeddedDimer_V=True, EmbeddedAtomNumber_V=EdEosMovingIndexAlongC)[0]
                            # EamDimerResult is: FileName,Lattice,BoxX1,BoxX2,BoxY1,BoxY2,BoxZ1,BoxZ2,id,type,x,y,z,c_eng,DimerDistance, RhoCalculated, FCalculated, PhiCalculated, EnergyTotal, EnergyNewPerAtom
                            # print("EamNew is:\n" + str(EamNew))
                            CostFuncQsdAlongCDumpNumpy = np.append(CostFuncQsdAlongCDumpNumpy, EamNew, axis=0)

            # print("EamAlongCNumpy is:\n" + str(EamAlongCNumpy))

            CostFuncQsdAlongCLogNumpy[:, 7] = CostFuncQsdAlongCLogNumpy[:, 7] - \
                                              np.min(CostFuncQsdAlongCLogNumpy, axis=0)[7]
            CostFuncQsdAlongCDumpNumpy[:, 19] = CostFuncQsdAlongCDumpNumpy[:, 19] - \
                                                np.min(CostFuncQsdAlongCDumpNumpy, axis=0)[19]
            # print("Min of EosMinNumpy is: " + str(np.min(EamNumpy,axis=0)[3]))
            # </editor-fold>

            # <editor-fold desc="AlongAc">
            if PrintSectionTitle_V: print("CostFunction-AlongAc")

            InitialPosX = EdEosInitialPosXAlongAc
            InitialPosY = EdEosInitialPosYAlongAc
            InitialPosZ = EdEosInitialPosZAlongAc

            CostFuncQsdAlongAcDumpNumpy = np.zeros((0, 20))
            CostFuncQsdAlongAcLogNumpy = np.zeros((0, 9))
            ErrorFree = True

            # print("EosMinNumpy is:\n" + str(EosMinNumpy))
            NameLatticeBoxConfigNumpyAll = np.zeros((0, 14))

            for root, dirs, files in os.walk(FolderAddressEamEdPpmAlongAc, topdown=False):  # open the files
                # print("root is: " + str(root))
                # print("files are: " + str(files))
                # print("dirs is: " + str(dirs))
                for name in files:
                    # print("File Name is: " + str(name))
                    if ".log" in name:  # target the dump files
                        # print("Log File opened: " + str(name))
                        LogFileAddress = os.path.join(root, name)
                        # LogExplorer(LogFileAddress_V, EmbeddedDimer_V = False, Along_V = "AlongAc", EOS_V = False):
                        EamLogNew = LogExplorer(LogFileAddress, EmbeddedDimer_V=True, Along_V="AlongAc", EOS_V=False,
                                                MirrorAtEq_V=True)
                        # print(EamLogNew)
                        if EamLogNew is False:
                            ErrorFree = False
                        else:
                            # print("EamLogNew is: " + str(EamLogNew))
                            # LogResult is ([[FileName, Length, BoxX, BoxY, BoxZ, NumOfAtoms, EvEnergy, EnergyPerAtom]])
                            CostFuncQsdAlongAcLogNumpy = np.append(CostFuncQsdAlongAcLogNumpy, EamLogNew,
                                                                   axis=0)
                            ErrorFree = True

                    elif ".dump" in name:  # target the dump files
                        # print("Dump File opened: " + str(name))
                        # print("root is: " + str(root))
                        # print(root.split("\\")[-1])
                        FolderName = int(root.split("\\")[-1])
                        # print("Folder is: " + str(FolderName))
                        if FolderName in range(ExtractingDumpRangeQsdEamAlongAcStart,
                                               ExtractingDumpRangeQsdEamAlongAcFinish):
                            # print("DUMP FILE OPENED: " + str(name))
                            # print(FileNameInRange)
                            DumpFileAddress = os.path.join(root, name)
                            # print(DumpFileAddress)
                            # EAM input is: (DumpFileAddress_V, CutOff_V, BondLenght_V, DistPhiSpline_V, RhoFSpline_V, DistVSpline_V, EmbeddedDimer_V = False, EmbeddedAtomNumber_V = 16, EOS = False)
                            EamNew = EamCalculator(DumpFileAddress, CutOff, DistRhoInterpolate,
                                                   RhoFInterpolate, DistPhiInterpolate,
                                                   BondLenght_V=BondLengthAlongAc,
                                                   EmbeddedDimer_V=True, EmbeddedAtomNumber_V=EdEosMovingIndexAlongAc,
                                                   MirrorAtEq_V=True)[0]
                            # EamDimerResult is: FileName,Lattice,BoxX1,BoxX2,BoxY1,BoxY2,BoxZ1,BoxZ2,id,type,x,y,z,c_eng,DimerDistance, RhoCalculated, FCalculated, PhiCalculated, EnergyTotal, EnergyNewPerAtom
                            # print("EamNew is:\n" + str(EamNew))
                            CostFuncQsdAlongAcDumpNumpy = np.append(CostFuncQsdAlongAcDumpNumpy, EamNew, axis=0)

            # print("EamAlongAcNumpy is:\n" + str(EamAlongAcNumpy))

            CostFuncQsdAlongAcLogNumpy[:, 7] = CostFuncQsdAlongAcLogNumpy[:, 7] - \
                                               np.min(CostFuncQsdAlongAcLogNumpy, axis=0)[7]
            CostFuncQsdAlongAcDumpNumpy[:, 19] = CostFuncQsdAlongAcDumpNumpy[:, 19] - \
                                                 np.min(CostFuncQsdAlongAcDumpNumpy, axis=0)[19]
            # print("Min of EosMinNumpy is: " + str(np.min(EamNumpy,axis=0)[3]))
            # </editor-fold>

            # <editor-fold desc="^^^^^ Exporting">
            print("^^^^^ Exporting")
            if DeepAnalysis_V:
                np.savetxt(Date + "-CostFunc-" + StageName_V + "--" + StageName_V + "-QsdAlongALogNumpy.csv",
                           CostFuncQsdAlongALogNumpy, delimiter=",",
                           header="FileName,Length,BoxX,BoxY,BoxZ,NumOfAtoms,EvEnergy,EnergyPerAtom", comments="")
                np.savetxt(Date + "-CostFunc-" + StageName_V + "-QsdAlongADumpNumpy.csv", CostFuncQsdAlongADumpNumpy,
                           delimiter=",",
                           header="FileName,Lattice,BoxX1,BoxX2,BoxY1,BoxY2,BoxZ1,BoxZ2,id,type,x,y,z,c_eng,LatticeSizeNew,RhoCalculated,FCalculated,PhiCalculated,EnergyTotal,EnergyNewPerAtom",
                           comments="")

                np.savetxt(Date + "-CostFunc-" + StageName_V + "-QsdAlongCLogNumpy.csv", CostFuncQsdAlongCLogNumpy,
                           delimiter=",",
                           header="FileName,Length,BoxX,BoxY,BoxZ,NumOfAtoms,EvEnergy,EnergyPerAtom", comments="")
                np.savetxt(Date + "-CostFunc-" + StageName_V + "-QsdAlongCDumpNumpy.csv", CostFuncQsdAlongCDumpNumpy,
                           delimiter=",",
                           header="FileName,Lattice,BoxX1,BoxX2,BoxY1,BoxY2,BoxZ1,BoxZ2,id,type,x,y,z,c_eng,LatticeSizeNew,RhoCalculated,FCalculated,PhiCalculated,EnergyTotal,EnergyNewPerAtom",
                           comments="")

                np.savetxt(Date + "-CostFunc-" + StageName_V + "-QsdAlongAcLogNumpy.csv", CostFuncQsdAlongAcLogNumpy,
                           delimiter=",",
                           header="FileName,Length,BoxX,BoxY,BoxZ,NumOfAtoms,EvEnergy,EnergyPerAtom", comments="")
                np.savetxt(Date + "-CostFunc-" + StageName_V + "-QsdAlongAcDumpNumpy.csv", CostFuncQsdAlongAcDumpNumpy,
                           delimiter=",",
                           header="FileName,Lattice,BoxX1,BoxX2,BoxY1,BoxY2,BoxZ1,BoxZ2,id,type,x,y,z,c_eng,LatticeSizeNew,RhoCalculated,FCalculated,PhiCalculated,EnergyTotal,EnergyNewPerAtom",
                           comments="")

            # </editor-fold>

        elif EamSource_V == "RunExtractExternalLoop":
            InternalLoop = False
            WorkingDirectory = CostFunctionDirectory

            # <editor-fold desc="Run">
            if PrintSectionTitle_V: print("CostFunction-Run")

            for Direction in DirectionList_V:
                # print(Direction)
                DirectionDirectory = WorkingDirectory + "/" + Direction
                Path(DirectionDirectory).mkdir(parents=True, exist_ok=True)

                RunLammps = DicRunLammpsRange[Direction]
                RunLammpsStart = RunLammps["Step"]["Start"]
                RunLammpsFinish = RunLammps["Step"]["Finish"]
                # print("Range is: " + str(RunLammpsStart) + " " + str(RunLammpsFinish))

                LammpsTempFileName = QsdTemplate + Direction + ".lammpstemp"

                if MultiProcessor_V:
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        result = [executor.submit(RunLammps, Num) for Num in range(RunLammpsStart, RunLammpsFinish)]
                else:
                    if RunLammpsOnlyCostRange_V:
                        CostRange = DicCostRange[Direction].to_numpy()[:, 0]
                    else:
                        CostRange = range(RunLammpsStart, RunLammpsFinish)

                    for CostPoint in CostRange:
                        LammpsTempFileAddress = LammpsTempAddress_V + "/" + LammpsTempFileName
                        LammpsInFileName = str(CostPoint) + ".in"
                        LammpsInFileAddress = DirectionDirectory + "/" + LammpsInFileName

                        shutil.copyfile(LammpsTempFileAddress, LammpsInFileAddress)
                        # shutil.copyfile(TestDirectory + "/" + PotentialName + "_ASE.eampot", DimerDirectory + "/" + PotentialFileName)
                        # shutil.copyfile(TestDirectory + "/" + "0_100000.restart", DimerDirectory + "/" + "0_100000.restart")

                        with fileinput.FileInput(LammpsInFileAddress, inplace=True) as file:
                            for line in file:
                                line = line.replace("WorkingDirectoryTemp", WorkingDirectory)
                                line = line.replace("PotentialTemp", PotentialFileAddress)
                                line = line.replace("NumTemp", str(CostPoint))
                                line = line.replace("CurrentDirectoryTemp", CurrentDirectory)
                                line = line.replace("DirectionDirectoryTemp", DirectionDirectory)
                                line = line.replace("StageNameTemp", str(StageName_V))
                                if RunLammpsEph_V:
                                    line = line.replace("RunLammpsEphTemp ", "")
                                else:
                                    line = line.replace("RunLammpsEphTemp ", "#")
                                if WriteLog_V:
                                    line = line.replace("WriteLogTemp ", "")
                                else:
                                    line = line.replace("WriteLogTemp ", "#")
                                if WriteDump_V:
                                    line = line.replace("WriteDumpTemp ", "")
                                else:
                                    line = line.replace("WriteDumpTemp ", "#")
                                if WriteReport_V:
                                    line = line.replace("WriteReportTemp ", "")
                                else:
                                    line = line.replace("WriteReportTemp ", "#")
                                # print(line, end="")

                        LammpsArgs = ["-screen", LammpsScreen]
                        lmp = lammps(cmdargs=LammpsArgs)
                        lmp.file(LammpsInFileAddress)
                        lmp.close()
            # </editor-fold>

            # <editor-fold desc="Extraction">
            if PrintSectionTitle_V: print("CostFunction-Extraction")
            for Direction in DirectionList_V:
                DirectionDirectory = WorkingDirectory + "/" + Direction

                Rev = DicRev[Direction]

                EamNumpy = np.zeros((0, 2))
                ErrorFree = True

                for root, dirs, files in os.walk(DirectionDirectory, topdown=False):  # open the files
                    # print("root is: " + str(root))
                    # print("files are: " + str(files))
                    # print("dirs is: " + str(dirs))
                    for name in files:
                        # print("File Name is: " + str(name))
                        if ReadEamEnergyFrom_V in name:  # target the dump files
                            # print("Log File opened: " + str(name))
                            # def FunctionSelector(FunctionType_V, Direction_V, ReadEamEnergyFrom_V, RunLammpsEph_V,DistRhoInterpolate_V, RhoFInterpolate_V, DistPhiInterpolate_V, SameDirectory_V, MirrorAtEq_V):
                            FunctionSelectorResult = FunctionSelector(FunctionType_V, Direction, ReadEamEnergyFrom_V,
                                                                      MeasureRho_V, DistRhoInterpolate,
                                                                      RhoFInterpolate, DistPhiInterpolate,
                                                                      SameDirectory_V, MirrorAtEq_V,DicRev)
                            # print(FunctionSelectorResult)
                            EamMeasuringFunction = FunctionSelectorResult[0]
                            EamMeasuringFunctionArgs = FunctionSelectorResult[1]
                            EamMeasuringFunctionColumnDistance = FunctionSelectorResult[3]
                            EamMeasuringFunctionColumnEnergy = FunctionSelectorResult[4]
                            # print(EamMeasuringFunctionColumnEnergy)
                            FileAddress = os.path.join(root, name)
                            EamNumpyNew = EamMeasuringFunction(FileAddress, *EamMeasuringFunctionArgs)
                            # print(EamNumpyNew)
                            EamNumpyNew = EamNumpyNew[:,
                                          [EamMeasuringFunctionColumnDistance, EamMeasuringFunctionColumnEnergy]]
                            # print(EamNumpyNew)

                            if EamNumpyNew is False:
                                ErrorFree = False
                            else:
                                # print("EamLogNew is: " + str(EamLogNew))
                                # LogResult is ([[FileName, Length, BoxX, BoxY, BoxZ, NumOfAtoms, EvEnergy, EnergyPerAtom]])
                                EamNumpy = np.append(EamNumpy, EamNumpyNew, axis=0)
                                ErrorFree = True

                # print(EamNumpy)
                EamNumpy[:, 1] = EamNumpy[:, 1] - np.min(EamNumpy, axis=0)[1]
                EamDf = pd.DataFrame(EamNumpy, columns=["Dist", "Y"])
                EamDf["Type"] = Direction
                # EamDf["TimesCalled"] = TimesCalled
                print("Here")
                print(EamDf)
            # </editor-fold>

            # <editor-fold desc="Exporting">
            if PrintSectionTitle_V: print("CostFunction-Exporting")
            if DeepAnalysis_V:
                EamDf.to_csv(
                    WorkingDirectory + "/" + Date + "-CostFunc-" + StageName_V + "-EamNumpy-" + str(TimesCalled) + ".csv",
                    index=False)
            # </editor-fold>

            # </editor-fold>

        elif EamSource_V == "RunExtractInternalLoop":
            WorkingDirectory = CostFunctionDirectory
            InternalLoop = True

            # <editor-fold desc="-- Run">
            if PrintSectionTitle_V: print("-- Run")
            if TimeRecord_V: start_time_L2 = time.time()
            for Direction in DirectionList_V:
                # <editor-fold desc="Writing Input File">
                if PrintSectionTitle_V: print("Writing Input File")
                if TimeRecord_V: start_time_L3 = time.time()
                # print(Direction)

                if RunLammpsOnlyCostRange_V:
                    CostRange = DicCostRange[Direction]["Step"]
                    RunLammpsStart = CostRange.min() - 1
                    RunLammpsFinish = CostRange.max() + 1
                else:
                    RunLammps = DicRunLammpsRange[Direction]
                    RunLammpsStart = RunLammps["Step"]["Start"]
                    RunLammpsFinish = RunLammps["Step"]["Finish"]
                    # print(LammpsInFileAddress)

                DirectionDirectory = WorkingDirectory + "/" + Direction
                Path(DirectionDirectory).mkdir(parents=True, exist_ok=True)
                LammpsTempFileName = QsdTemplate + Direction + ".lammpstemp"
                LammpsTempFileAddress = LammpsTempAddress_V + "/" + LammpsTempFileName
                LammpsInFileName = str(TimesCalled) + ".in"
                LammpsInFileAddress = DirectionDirectory + "/" + LammpsInFileName
                # print(LammpsTempFileAddress)
                # print(LammpsInFileAddress)
                shutil.copyfile(LammpsTempFileAddress, LammpsInFileAddress)
                # shutil.copyfile(TestDirectory + "/" + PotentialName + "_ASE.eampot", DimerDirectory + "/" + PotentialFileName)
                # shutil.copyfile(TestDirectory + "/" + "0_100000.restart", DimerDirectory + "/" + "0_100000.restart")

                TempToInput(LammpsInFileAddress, WorkingDirectory, PotentialFileAddress, CurrentDirectory,
                            DirectionDirectory,
                            StageName, TimesCalled, RhoFinderName_V,
                            InternalLoop, RunLammpsStart, RunLammpsFinish,
                            RunLammpsEph_V, WriteLog_V, WriteDump_V,
                            WriteReport_V,MeasureRho_V, MeasureDFer_V)

                if TimeRecord_V: print("___________________" + str(time.time() - start_time_L3),
                                       " seconds"); start_time_L3 = time.time()
                # </editor-fold>

                # <editor-fold desc="Call Lammps">
                if PrintSectionTitle_V: print("Call Lammps")
                if TimeRecord_V: start_time_L3 = time.time()
                LammpsArgs = ["-screen", LammpsScreen]
                lmp = lammps(cmdargs=LammpsArgs)
                lmp.file(LammpsInFileAddress)
                lmp.close()
                if TimeRecord_V: print("___________________" + str(time.time() - start_time_L3),
                                       " seconds"); start_time_L3 = time.time()
                # </editor-fold>
            if TimeRecord_V: print("____________________________________" + str(time.time() - start_time_L2),
                                   " seconds"); start_time_L2 = time.time()
            # </editor-fold>

            # <editor-fold desc="-- Extraction">
            if PrintSectionTitle_V: print("-- Extraction")
            if TimeRecord_V: start_time_L2 = time.time()
            if (MeasureRho_V) or (RunLammpsEph_V) :
                EamDf = pd.DataFrame(columns=["Type", "Dist", "Rho", "Energy"])
            else:
                EamDf = pd.DataFrame(columns=["Type", "Dist", "Energy"])
            # print(DirectionList_V)
            for Direction in DirectionList_V:
                DirectionDirectory = WorkingDirectory + "/" + Direction
                Rev = DicRev[Direction]

                if (MeasureRho_V) or (RunLammpsEph_V):
                    EamNumpy = np.zeros((0, 3))
                else:
                    EamNumpy = np.zeros((0, 2))

                ErrorFree = True

                for root, dirs, files in os.walk(DirectionDirectory, topdown=False):  # open the files
                    # print("root is: " + str(root))
                    # print("files are: " + str(files))
                    # print("dirs is: " + str(dirs))
                    for name in files:
                        # print("File Name is: " + str(name))
                        if ReadEamEnergyFrom_V in name:  # target the dump files
                            # print("Log File opened: " + str(name))
                            # def FunctionSelector(FunctionType_V, Direction_V, ReadEamEnergyFrom_V, RunLammpsEph_V,DistRhoInterpolate_V, RhoFInterpolate_V, DistPhiInterpolate_V, SameDirectory_V, MirrorAtEq_V):

                            FunctionSelectorResult = FunctionSelector(FunctionType_V, Direction, ReadEamEnergyFrom_V,
                                                                      MeasureRho_V, DistRhoInterpolate,
                                                                      RhoFInterpolate, DistPhiInterpolate,
                                                                      SameDirectory_V, MirrorAtEq_V,
                                                                      SkipHeader,DicRev)
                            # print(FunctionSelectorResult)
                            EamMeasuringFunction = FunctionSelectorResult[0]
                            EamMeasuringFunctionArgs = FunctionSelectorResult[1]
                            EamMeasuringFunctionColumnDistance = FunctionSelectorResult[3]
                            # print(EamMeasuringFunctionColumnDistance)
                            EamMeasuringFunctionColumnEnergy = FunctionSelectorResult[4]
                            # print(EamMeasuringFunctionColumnEnergy)
                            EamMeasuringFunctionColumnRho = FunctionSelectorResult[5]
                            FileAddress = os.path.join(root, name)
                            EamNumpyNew = EamMeasuringFunction(FileAddress, *EamMeasuringFunctionArgs)
                            # print(EamNumpyNew)
                            if (MeasureRho_V) or (RunLammpsEph_V):
                                EamNumpyNew = EamNumpyNew[:,[EamMeasuringFunctionColumnDistance, EamMeasuringFunctionColumnRho, EamMeasuringFunctionColumnEnergy]]
                            else:
                                EamNumpyNew = EamNumpyNew[:,[EamMeasuringFunctionColumnDistance, EamMeasuringFunctionColumnEnergy]]
                            # print(EamNumpyNew)

                            if EamNumpyNew is False:
                                ErrorFree = False
                            else:
                                # print("EamLogNew is: " + str(EamLogNew))
                                # LogResult is ([[FileName, Length, BoxX, BoxY, BoxZ, NumOfAtoms, EvEnergy, EnergyPerAtom]])
                                EamNumpy = np.append(EamNumpy, EamNumpyNew, axis=0)
                                ErrorFree = True
                # print(EamNumpy)
                if (MeasureRho_V) or (RunLammpsEph_V):
                    EamDfDirection = pd.DataFrame(EamNumpy, columns=["Dist", "Rho", "Energy"])
                else:
                    EamDfDirection = pd.DataFrame(EamNumpy, columns=["Dist", "Energy"])

                EamDfDirection["Energy"] = EamDfDirection["Energy"] - EamDfDirection["Energy"].min()
                EamDfDirection["Type"] = Direction
                if (MeasureRho_V) or (RunLammpsEph_V):
                    EamDf["F"] = RhoFInterpolate(EamDf["Rho"])
                    EamDf["Phi"] = 2 * (EamDf["Energy"] - EamDf["F"])
                # EamDfDirection["TimesCalled"] = TimesCalled
                # print(EamDfDirection)
                EamDf = pd.concat((EamDf, EamDfDirection), ignore_index=True)
                # print(EamDf)

                # </editor-fold>
            EamDfGrouped = EamDf.groupby(by=["Type"])
            if TimeRecord_V: print("____________________________________" + str(time.time() - start_time_L2),
                                   " seconds"); start_time_L2 = time.time()
            # </editor-fold>

        elif EamSource_V == "Read":
            # <editor-fold desc="EXTRACTING">
            if PrintSectionTitle_V: print("CostFunction-EXTRACTING")
            WorkingDirectory = CostFunctionDirectory
            CostFuncQsdAlongALogNumpy = np.genfromtxt(
                WorkingDirectory + "/" + Date + "-CostFunc-" + StageName_V + "-QsdAlongALogNumpy.csv",
                delimiter=",", skip_header=1)
            CostFuncQsdAlongCLogNumpy = np.genfromtxt(
                WorkingDirectory + "/" + Date + "-CostFunc-" + StageName_V + "-QsdAlongCLogNumpy.csv",
                delimiter=",", skip_header=1)
            CostFuncQsdAlongAcLogNumpy = np.genfromtxt(
                WorkingDirectory + "/" + Date + "-CostFunc-" + StageName_V + "-QsdAlongAcLogNumpy.csv",
                delimiter=",", skip_header=1)
            if ReadEamEnergyFrom_V == "Dump":
                CostFuncQsdAlongADumpNumpy = np.genfromtxt(Date + "-CostFunc-" + StageName_V + "-QsdAlongADumpNumpy.csv",
                                                           delimiter=",", skip_header=1)
                CostFuncQsdAlongCDumpNumpy = np.genfromtxt(Date + "-CostFunc-" + StageName_V + "-QsdAlongCDumpNumpy.csv",
                                                           delimiter=",", skip_header=1)
                CostFuncQsdAlongAcDumpNumpy = np.genfromtxt(
                    Date + "-CostFunc-" + StageName_V + "-QsdAlongAcDumpNumpy.csv",
                    delimiter=",", skip_header=1)

            # print("Size of EosEamEgtSatZblZblPpmLogNumpy is:")
            # print(EosEamEgtSatZblZblPpmLogNumpy.shape)
            # print("\nEosEamOriginalLogNumpy is:")
            # print(EosEamOriginalLogNumpy)
            # print("Size of EosEamEgtSatZblZblPpmDumpNumpy is:")
            # print(EosEamEgtSatZblZblPpmDumpNumpy.shape)
            # print("\nCaDf is:")
            # print(CaDf)
            # </editor-fold>

        # </editor-fold>
    elif FunctionType_V == "Eos":
        # <editor-fold desc="^^^^^ Eos">
        if PrintSectionTitle_V: print("^^^^^ Eos")
        if EamSource_V == "Extract":
            # <editor-fold desc="FunctionExtract">
            print("EXTRACTING")
            CostFuncEosLogNumpy = np.zeros((0, 9))
            CostFuncEosDumpNumpy = np.zeros((0, 20))
            ErrorFree = True

            for root, dirs, files in os.walk(FolderAddressEosEamPpm, topdown=False):  # open the files
                # print("root is: " + str(root))
                # print("files are: " + str(files))
                # print("dirs is: " + str(dirs))
                for name in files:
                    # print("File Name is: " + str(name))
                    if ".log" in name:  # target the dump files
                        # print("Log File opened: " + str(name))
                        LogFileAddress = os.path.join(root, name)
                        EamLogNew = LogExplorer(LogFileAddress, EOS_V=True)
                        # print(EamLogNew)
                        if EamLogNew is False:
                            ErrorFree = False
                        else:
                            # print("EamLogNew is: " + str(EamLogNew))
                            # LogResult is ([[FileName, Length, BoxX, BoxY, BoxZ, NumOfAtoms, EvEnergy, EnergyPerAtom]])
                            CostFuncEosLogNumpy = np.append(CostFuncEosLogNumpy, EamLogNew, axis=0)
                            ErrorFree = True

                    elif ".dump" in name:  # target the dump files
                        # print("Dump File opened: " + str(name))
                        FileNameInRange = int(root.split("\\")[-1])
                        if FileNameInRange in range(-22, 50):
                            # print("DUMP FILE OPENED: " + str(name))
                            # print(FileNameInRange)
                            DumpFileAddress = os.path.join(root, name)
                            # print(DumpFileAddress)
                            # def EamCalculator (DumpFileAddress_V, CutOff_V, BondLenght_V, DistPhiSpline_V, RhoFSpline_V, DistVSpline_V, EmbeddedDimer_V = False, EmbeddedAtomNumber_V = 16, EOS_V = False):
                            EamNew = EamCalculator(DumpFileAddress, CutOff,
                                                   DistRhoInterpolate, RhoFInterpolate,
                                                   DistPhiInterpolate,
                                                   EOS_V=True)[0]
                            # EamEosResult is: FileName,Lattice,BoxX1,BoxX2,BoxY1,BoxY2,BoxZ1,BoxZ2,id,type,x,y,z,c_eng,LatticeSizeNew,RhoCalculated,FCalculated,PhiCalculated,EnergyTotal,EnergyNewPerAtom
                            # print("EamNew is:\n" + str(EamNew))
                            CostFuncEosDumpNumpy = np.append(CostFuncEosDumpNumpy, EamNew, axis=0)

            # print(CostFuncEosLogNumpy)
            # print(np.min(CostFuncEosLogNumpy,axis=0)[7])
            CostFuncEosLogNumpy[:, 7] = CostFuncEosLogNumpy[:, 7] - \
                                        np.min(CostFuncEosLogNumpy, axis=0)[7]

            CostFuncEosDumpNumpy[:, 19] = CostFuncEosDumpNumpy[:, 19] - \
                                          np.min(CostFuncEosDumpNumpy, axis=0)[19]
            # </editor-fold>

            # <editor-fold desc="Exporting">
            print("Exporting")
            np.savetxt(Date + "-" + StageName + "-ExtractingCostFuncEosLogNumpy.csv", CostFuncEosLogNumpy,
                       delimiter=",",
                       header="FileName,Length,BoxX,BoxY,BoxZ,NumOfAtoms,EvEnergy,EnergyPerAtom", comments="")
            np.savetxt(Date + "-" + StageName + "-ExtractingCostFuncEosDumpNumpy.csv", CostFuncEosDumpNumpy,
                       delimiter=",",
                       header="FileName,Lattice,BoxX1,BoxX2,BoxY1,BoxY2,BoxZ1,BoxZ2,id,type,x,y,z,c_eng,LatticeSizeNew,RhoCalculated,FCalculated,PhiCalculated,EnergyTotal,EnergyNewPerAtom",
                       comments="")
            # </editor-fold>

        elif EamSource_V == "RunExtractExternalLoop":

            # <editor-fold desc="-- Run">
            if PrintSectionTitle_V: print("CostFunction-Run")

            WorkingDirectory = CostFunctionDirectory

            Path(WorkingDirectory).mkdir(parents=True, exist_ok=True)
            # print(WorkingDirectory)

            RunLammps = DicRunLammpsRange[FunctionType_V]
            RunLammpsStart = RunLammps["Step"]["Start"]
            RunLammpsFinish = RunLammps["Step"]["Finish"]

            LammpsTempFileName = EosTemplate

            if MultiProcessor_V:
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    result = [executor.submit(RunLammps, Num) for Num in range(RunLammpsStart, RunLammpsFinish)]

            else:
                if RunLammpsOnlyCostRange_V:
                    CostRange = DicCostRange[FunctionType_V].to_numpy()[:, 0]
                else:
                    CostRange = range(RunLammpsStart, RunLammpsFinish)

                for CostPoint in CostRange:
                    LammpsTempFileAddress = LammpsTempAddress_V + "/" + LammpsTempFileName
                    LammpsInFileName = str(CostPoint) + ".in"
                    LammpsInFileAddress = WorkingDirectory + "/" + LammpsInFileName
                    # print(LammpsInFileAddress)

                    shutil.copyfile(LammpsTempFileAddress, LammpsInFileAddress)
                    with fileinput.FileInput(LammpsInFileAddress, inplace=True) as file:
                        for line in file:
                            line = line.replace("WorkingDirectoryTemp", WorkingDirectory)
                            line = line.replace("PotentialFileAddressTemp", PotentialFileAddress)
                            line = line.replace("NumTemp", str(CostPoint))
                            line = line.replace("CurrentDirectoryTemp", CurrentDirectory)
                            line = line.replace("StageNameTemp", str(StageName_V))
                            line = line.replace("RhoFinderNameTemp", str(RhoFinderName_V))
                            line = line.replace("ExternalLoopTemp ", "")
                            line = line.replace("InternalLoopTemp ", "#")
                            line = line.replace("RunLammpsStartTemp", str(RunLammpsStart))
                            line = line.replace("RunLammpsFinishTemp", str(RunLammpsFinish))
                            if RunLammpsEph_V:
                                line = line.replace("RunLammpsEphTemp ", "")
                            else:
                                line = line.replace("RunLammpsEphTemp ", "#")
                            if WriteLog_V:
                                line = line.replace("WriteLogTemp ", "")
                            else:
                                line = line.replace("WriteLogTemp ", "#")
                            if WriteDump_V:
                                line = line.replace("WriteDumpTemp ", "")
                            else:
                                line = line.replace("WriteDumpTemp ", "#")
                            if WriteReport_V:
                                line = line.replace("WriteReportTemp ", "")
                            else:
                                line = line.replace("WriteReportTemp ", "#")
                            print(line, end="")

                    LammpsArgs = ["-screen", LammpsScreen]
                    lmp = lammps(cmdargs=LammpsArgs)
                    lmp.file(LammpsInFileAddress)
                    lmp.close()

            # </editor-fold>

            # <editor-fold desc="-- EXTRACTING">
            if PrintSectionTitle_V: print("CostFunction-EXTRACTING")
            EamNumpy = np.zeros((0, 2))
            ErrorFree = True

            for root, dirs, files in os.walk(WorkingDirectory, topdown=False):  # open the files
                # print("root is: " + str(root))
                # print("files are: " + str(files))
                # print("dirs is: " + str(dirs))
                for name in files:
                    # print("File Name is: " + str(name))
                    if ReadEamEnergyFrom_V in name:  # target the dump files
                        # print("Log File opened: " + str(name))
                        # def FunctionSelector(FunctionType_V, Direction_V, ReadEamEnergyFrom_V, RunLammpsEph_V,DistRhoInterpolate_V, RhoFInterpolate_V, DistPhiInterpolate_V,SameDirectory_V, MirrorAtEq_V
                        FunctionSelectorResult = FunctionSelector(FunctionType_V, "", ReadEamEnergyFrom_V,
                                                                  MeasureRho_V, DistRhoInterpolate, RhoFInterpolate,
                                                                  DistPhiInterpolate, SameDirectory_V, MirrorAtEq_V,
                                                                  SkipHeader)
                        EamMeasuringFunction = FunctionSelectorResult[0]
                        EamMeasuringFunctionArgs = FunctionSelectorResult[1]
                        EamMeasuringFunctionColumnDistance = FunctionSelectorResult[3]
                        EamMeasuringFunctionColumnEnergy = FunctionSelectorResult[4]
                        FileAddress = os.path.join(root, name)
                        FileAddress = os.path.join(root, name)
                        EamNumpyNew = EamMeasuringFunction(FileAddress, *EamMeasuringFunctionArgs)

                        # print(EamLogNew)
                        if EamNumpyNew is False:
                            ErrorFree = False
                        else:
                            # print("EamLogNew is: " + str(EamLogNew))
                            # LogResult is ([[FileName, Length, BoxX, BoxY, BoxZ, NumOfAtoms, EvEnergy, EnergyPerAtom]])
                            EamNumpyNew = EamNumpyNew[:,
                                          [EamMeasuringFunctionColumnDistance, EamMeasuringFunctionColumnEnergy]]
                            EamNumpy = np.append(EamNumpy, EamNumpyNew, axis=0)
                            ErrorFree = True
            EamNumpy[:, 1] = EamNumpy[:, 1] - np.min(EamNumpy, axis=0)[1]
            EamDf = pd.DataFrame(EamNumpy, columns=["Dist", "Y"])
            EamDf["Type"] = FunctionType_V
            # EamDf["TimesCalled"] = TimesCalled
            # </editor-fold>

            # <editor-fold desc="-- Exporting">
            if PrintSectionTitle_V: print("CostFunction-Exporting")
            if DeepAnalysis_V:
                EamDf.to_csv(
                    WorkingDirectory + "/" + Date + "-CostFunc-" + StageName_V + "-EamNumpy-" + str(TimesCalled) + ".csv",
                    index=False)
            # </editor-fold>

        elif EamSource_V == "RunExtractInternalLoop":
            InternalLoop = True

            # <editor-fold desc="-- Run">
            if PrintSectionTitle_V: print("-- Run")
            if TimeRecord_V: start_time_L2 = time.time()
            # <editor-fold desc="Writing Input File">
            if PrintSectionTitle_V: print("Writing Input File")
            if TimeRecord_V: start_time_L3 = time.time()

            if RunLammpsOnlyCostRange_V:
                CostRange = DicCostRange[FunctionType_V]["Step"]
                RunLammpsStart = CostRange.min() - 1
                RunLammpsFinish = CostRange.max() + 1
            else:
                RunLammps = DicRunLammpsRange[FunctionType_V]
                RunLammpsStart = RunLammps["Step"]["Start"]
                RunLammpsFinish = RunLammps["Step"]["Finish"]
                # print(LammpsInFileAddress)

            WorkingDirectory = CostFunctionDirectory
            Path(WorkingDirectory).mkdir(parents=True, exist_ok=True)
            DirectionDirectory = WorkingDirectory + "/"
            LammpsTempFileName = EosTemplate
            LammpsTempFileAddress = LammpsTempAddress_V + "/" + LammpsTempFileName
            LammpsInFileName = str(TimesCalled) + ".in"
            LammpsInFileAddress = WorkingDirectory + "/" + LammpsInFileName

            shutil.copyfile(LammpsTempFileAddress, LammpsInFileAddress)
            # shutil.copyfile(TestDirectory + "/" + PotentialName + "_ASE.eampot", DimerDirectory + "/" + PotentialFileName)
            # shutil.copyfile(TestDirectory + "/" + "0_100000.restart", DimerDirectory + "/" + "0_100000.restart")

            TempToInput(LammpsInFileAddress, WorkingDirectory, PotentialFileAddress, CurrentDirectory,
                        DirectionDirectory,
                        StageName, TimesCalled, RhoFinderName_V,
                        InternalLoop, RunLammpsStart, RunLammpsFinish,
                        RunLammpsEph_V, WriteLog_V, WriteDump_V,
                        WriteReport_V,MeasureRho_V, MeasureDFer_V)
            if TimeRecord_V: print("___________________" + str(time.time() - start_time_L3),
                                   " seconds"); start_time_L3 = time.time()
            # </editor-fold>

            # <editor-fold desc="Call Lammps">
            if PrintSectionTitle_V: print("Call Lammps")
            if TimeRecord_V: start_time_L3 = time.time()
            LammpsArgs = ["-screen", LammpsScreen]
            lmp = lammps(cmdargs=LammpsArgs)
            lmp.file(LammpsInFileAddress)
            lmp.close()
            if TimeRecord_V: print("___________________" + str(time.time() - start_time_L3),
                                   " seconds"); start_time_L3 = time.time()
            # </editor-fold>
            if TimeRecord_V: print("____________________________________" + str(time.time() - start_time_L2),
                                   " seconds"); start_time_L2 = time.time()
            # </editor-fold>

            # <editor-fold desc="-- EXTRACTING">
            if PrintSectionTitle_V: print("-- EXTRACTING")
            if TimeRecord_V: start_time_L2 = time.time()
            if RunLammpsEph_V:
                EamNumpy = np.zeros((0, 3))
            elif MeasureRho_V:
                EamNumpy = np.zeros((0, 4))
            else:
                EamNumpy = np.zeros((0, 2))
            ErrorFree = True

            for root, dirs, files in os.walk(WorkingDirectory, topdown=False):  # open the files
                # print("root is: " + str(root))
                # print("files are: " + str(files))
                # print("dirs is: " + str(dirs))
                for name in files:
                    # print("File Name is: " + str(name))
                    if ReadEamEnergyFrom_V in name:
                        # print("Log File opened: " + str(name))
                        # def FunctionSelector(FunctionType_V, Direction_V, ReadEamEnergyFrom_V, RunLammpsEph_V,DistRhoInterpolate_V, RhoFInterpolate_V, DistPhiInterpolate_V,SameDirectory_V, MirrorAtEq_V
                        FunctionSelectorResult = FunctionSelector(FunctionType_V, "", ReadEamEnergyFrom_V,
                                                                  MeasureRho_V, DistRhoInterpolate, RhoFInterpolate,
                                                                  DistPhiInterpolate, SameDirectory_V, MirrorAtEq_V,
                                                                  SkipHeader,DicRev)
                        EamMeasuringFunction = FunctionSelectorResult[0]
                        EamMeasuringFunctionArgs = FunctionSelectorResult[1]
                        EamMeasuringFunctionColumnDistance = FunctionSelectorResult[3]
                        EamMeasuringFunctionColumnEnergy = FunctionSelectorResult[4]
                        EamMeasuringFunctionColumnRho = FunctionSelectorResult[5]
                        EamMeasuringFunctionColumnFDer = FunctionSelectorResult[6]
                        FileAddress = os.path.join(root, name)
                        # print(FileAddress)
                        EamNumpyNew = EamMeasuringFunction(FileAddress, *EamMeasuringFunctionArgs)
                        # print(EamNumpyNew)

                        if RunLammpsEph_V:
                            EamNumpyNew = EamNumpyNew[:,[EamMeasuringFunctionColumnDistance,
                                                         EamMeasuringFunctionColumnRho,
                                                         EamMeasuringFunctionColumnEnergy]]
                        elif MeasureRho_V:
                            EamNumpyNew = EamNumpyNew[:,[EamMeasuringFunctionColumnDistance,
                                                         EamMeasuringFunctionColumnRho,
                                                         EamMeasuringFunctionColumnEnergy,
                                                         EamMeasuringFunctionColumnFDer]]
                        elif (MeasureRho_V) and (ReadEamEnergyFrom_V == ".dump"):
                            EamNumpyNew = EamNumpyNew[:, [EamMeasuringFunctionColumnDistance,
                                                          EamMeasuringFunctionColumnRho,
                                                          EamMeasuringFunctionColumnEnergy]]
                        else:
                            EamNumpyNew = EamNumpyNew[:,[EamMeasuringFunctionColumnDistance,
                                                         EamMeasuringFunctionColumnEnergy]]
                        # print(EamLogNew)

                        if EamNumpyNew is False:
                            ErrorFree = False
                        else:
                            # print("EamLogNew is: " + str(EamLogNew))
                            # LogResult is ([[FileName, Length, BoxX, BoxY, BoxZ, NumOfAtoms, EvEnergy, EnergyPerAtom]])
                            EamNumpy = np.append(EamNumpy, EamNumpyNew, axis=0)
                            ErrorFree = True

            if RunLammpsEph_V:
                EamDf = pd.DataFrame(EamNumpy, columns=["Dist", "Rho", "Energy"])
            elif MeasureRho_V:
                EamDf = pd.DataFrame(EamNumpy, columns=["Dist", "Rho", "Energy", "FDer"])
            elif (MeasureRho_V) and (ReadEamEnergyFrom_V == ".dump"):
                EamDf = pd.DataFrame(EamNumpy, columns=["Dist", "Rho", "Energy"])
            else:
                EamDf = pd.DataFrame(EamNumpy, columns=["Dist", "Energy"])

            EamDf["Energy"] = EamDf["Energy"] - EamDf["Energy"].min()
            EamDf["Type"] = FunctionType_V

            if (MeasureRho_V) or (RunLammpsEph_V):
                EamDf["F"] = RhoFInterpolate(EamDf["Rho"])
                EamDf["Phi"] = 2 * (EamDf["Energy"] - EamDf["F"])
            # EamDf["TimesCalled"] = TimesCalled

            EamDfGrouped = EamDf.groupby(by=["Type"])
            if TimeRecord_V: print("____________________________________" + str(time.time() - start_time_L2),
                                   " seconds"); start_time_L2 = time.time()
            # </editor-fold>

        elif EamSource_V == "Read":
            # <editor-fold desc="Read">
            WorkingDirectory = CostFunctionDirectory
            if PrintSectionTitle_V: print("CostFunction-Read")
            if ReadEamEnergyFrom_V == "Log":
                np.savetxt(WorkingDirectory + "/" + Date + "-CostFunc-EosLogNumpy.csv", CostFuncEosLogNumpy,
                           delimiter=",",
                           header="FileName,Length,BoxX,BoxY,BoxZ,NumOfAtoms,EvEnergy,EnergyPerAtom,Rho", comments="")
            if ReadEamEnergyFrom_V == "Dump":
                np.savetxt(WorkingDirectory + "/" + Date + "-CostFunc-EosDumpNumpy.csv", CostFuncEosDumpNumpy,
                           delimiter=",",
                           header="FileName,Lattice,BoxX1,BoxX2,BoxY1,BoxY2,BoxZ1,BoxZ2,id,type,x,y,z,c_eng,LatticeSizeNew,RhoCalculated,FCalculated,PhiCalculated,EnergyTotal,EnergyNewPerAtom",
                           comments="")
            if ReadEamEnergyFrom_V == "Report":
                np.savetxt(WorkingDirectory + "/" + Date + "-CostFunc-EosReportNumpy.csv", CostFuncEosReportNumpy,
                           delimiter=",",
                           header="Lattice,Step,Time,dt,Cpu,TemperatureIonic,Press,EnergyPotential,EnergyKinetic",
                           comments="")

            # print("Size of CostFuncEosLogNumpy is:")
            # print(CostFuncEosLogNumpy.shape)
            # print("\nEosEamOriginalLogNumpy is:")
            # print(EosEamOriginalLogNumpy)
            # print("Size of CostFuncEosDumpNumpy is:")
            # print(CostFuncEosDumpNumpy.shape)
            # print("\nCaDf is:")
            # print(CaDf)
            # </editor-fold>
        # </editor-fold>
        # </editor-fold>
    # </editor-fold>
    if TimeRecord_V: print("________________________________________________________" + str(
        time.time() - start_time_L1) + "seconds"); start_time_L1 = time.time()
    # </editor-fold>

    # <editor-fold desc="**********  Variable VS. EAM Functions">
    if PrintSectionTitle_V: print("**********  Variable VS. EAM Functions")
    # <editor-fold desc="$$$$$ Type Dependency">
    if FunctionType_V == "Qsd":
        for Direction in DirectionList_V:

            # <editor-fold desc="Ungrouping">
            EamDfGroup = EamDfGrouped.get_group(Direction).reset_index(drop=True)
            EamDfGroup = EamDfGroup.sort_values(by=["Dist"])
            # </editor-fold>

            # <editor-fold desc="DicEamDfStage">
            EamDfGroupDist = EamDfGroup["Dist"].reset_index(drop=True)
            EamDfGroupEnergy = EamDfGroup["Energy"].reset_index(drop=True)
            DicEamDfStage[StageName][Direction]["Dist"] = EamDfGroupDist
            DicEamDfStage[StageName][Direction]["DistEnergy"] = EamDfGroupEnergy
            # </editor-fold>

            # <editor-fold desc="Min and Max">
            EamDfGroupDistMin = EamDfGroupDist.min()
            EamDfGroupDistMax = EamDfGroupDist.max()
            # </editor-fold>

            # <editor-fold desc="DicEamDfInterpolateValueStage">
            EamDfGridDist = np.mgrid[EamDfGroupDistMin:EamDfGroupDistMax:dr]
            DicEamDfInterpolateValueStage[StageName_V][Direction]["Dist"] = EamDfGridDist
            # </editor-fold>

            # <editor-fold desc="DicRunLammpsRange">
            DicRunLammpsRange[Direction]["Dist"]["Start"] = EamDfGroupDistMin
            DicRunLammpsRange[Direction]["Dist"]["Finish"] = EamDfGroupDistMax
            # </editor-fold>

            if (MeasureRho_V) or (RunLammpsEph_V):

                # <editor-fold desc="DicEamDfStage">
                EamDfGroupRho = EamDfGroup["Rho"].reset_index(drop=True)
                EamDfGroupEnergy = EamDfGroup["Energy"].reset_index(drop=True)
                DicEamDfStage[StageName][Direction]["Rho"] = EamDfGroupRho
                DicEamDfStage[StageName][Direction]["RhoEnergy"] = EamDfGroupEnergy

                if DeepAnalysis_V:
                    EamDfGroupF = EamDfGroup["F"].reset_index(drop=True)
                    EamDfGroupPhi = EamDfGroup["Phi"].reset_index(drop=True)
                    DicEamDfStage[StageName][Direction]["F"] = EamDfGroupF
                    DicEamDfStage[StageName][Direction]["Phi"] = EamDfGroupPhi
                # </editor-fold>

                # <editor-fold desc="Min and Max">
                EamDfRhoMin = EamDfGroupRho.min()
                EamDfRhoMax = EamDfGroupRho.max()
                # </editor-fold>

                # <editor-fold desc="DicEamDfInterpolateValueStage">
                EamDfGrid = np.mgrid[EamDfRhoMin:EamDfRhoMax:drho]
                DicEamDfInterpolateValueStage[StageName_V][Direction]["Rho"] = EamDfGrid
                # </editor-fold>

                # <editor-fold desc="DicRunLammpsRange">
                DicRunLammpsRange[Direction]["Rho"]["Start"] = EamDfRhoMin
                DicRunLammpsRange[Direction]["Rho"]["Finish"] = EamDfRhoMax
                # </editor-fold>

                if MeasureRho_V:
                    if DeepAnalysis_V:
                        # <editor-fold desc="DicEamDfStage">
                        EamDfGroupFDer = EamDfGroup["FDer"]
                        DicEamDfStage[StageName][Direction]["FDer"] = EamDfGroupFDer
                        # </editor-fold>

    elif FunctionType_V == "Eos":

        # <editor-fold desc="Ungrouping">
        EamDfGroup = EamDfGrouped.get_group(FunctionType_V).reset_index(drop=True)
        EamDfGroup = EamDfGroup.sort_values(by=["Dist"])
        # </editor-fold>

        # <editor-fold desc="DicEamDfStage">
        EamDfGroupDist = EamDfGroup["Dist"].reset_index(drop=True)
        EamDfGroupEnergy = EamDfGroup["Energy"].reset_index(drop=True)
        DicEamDfStage[StageName][FunctionType_V]["Dist"] = EamDfGroupDist
        DicEamDfStage[StageName][FunctionType_V]["DistEnergy"] = EamDfGroupEnergy
        # </editor-fold>

        # <editor-fold desc="Min and Max">
        EamDfGroupDistMin = EamDfGroupDist.min()
        EamDfGroupDistMax = EamDfGroupDist.max()
        # </editor-fold>

        # <editor-fold desc="DicEamDfInterpolateValueStage">
        EamDfGridDist = np.mgrid[EamDfGroupDistMin:EamDfGroupDistMax:dr]
        DicEamDfInterpolateValueStage[StageName_V][FunctionType_V]["Dist"] = EamDfGridDist
        # </editor-fold>

        # <editor-fold desc="DicRunLammpsRange">
        DicRunLammpsRange[FunctionType_V]["Dist"]["Start"] = EamDfGroupDistMin
        DicRunLammpsRange[FunctionType_V]["Dist"]["Finish"] = EamDfGroupDistMax
        # </editor-fold>

        if (MeasureRho_V) or (RunLammpsEph_V):

            # <editor-fold desc="DicEamDfStage">
            EamDfGroupRho = EamDfGroup["Rho"].reset_index(drop=True)
            EamDfGroupEnergy = EamDfGroup["Energy"].reset_index(drop=True)
            DicEamDfStage[StageName][FunctionType_V]["Rho"] = EamDfGroupRho
            DicEamDfStage[StageName][FunctionType_V]["RhoEnergy"] = EamDfGroupEnergy

            if DeepAnalysis_V:
                EamDfGroupF = EamDfGroup["F"].reset_index(drop=True)
                EamDfGroupPhi = EamDfGroup["Phi"].reset_index(drop=True)
                DicEamDfStage[StageName][FunctionType_V]["F"] = EamDfGroupF
                DicEamDfStage[StageName][FunctionType_V]["Phi"] = EamDfGroupPhi
            # </editor-fold>

            # <editor-fold desc="Min and Max">
            EamDfRhoMin = EamDfGroupRho.min()
            EamDfRhoMax = EamDfGroupRho.max()
            # </editor-fold>

            # <editor-fold desc="DicEamDfInterpolateValueStage">
            EamDfGridRho = np.mgrid[EamDfRhoMin:EamDfRhoMax:drho]
            DicEamDfInterpolateValueStage[StageName_V][FunctionType_V]["Rho"] = EamDfGridRho
            # </editor-fold>

            # <editor-fold desc="DicRunLammpsRange">
            DicRunLammpsRange[FunctionType_V]["Rho"]["Start"] = EamDfRhoMin
            DicRunLammpsRange[FunctionType_V]["Rho"]["Finish"] = EamDfRhoMax
            # </editor-fold>

            # <editor-fold desc="Interpolate Functions">
            if DeepAnalysis_V:
                EamDfDistFInterpolate = PchipFunc(EamDfGroup["Dist"], EamDfGroup["F"])
                # add_boundary_knots(EamDfDistFInterpolate, 1, 2)
                EamDfDistFInterpolateValue = PchipValue(EamDfGroup["Dist"], EamDfGroup["F"], EamDfGridDist)
                DicEamDfInterpolateStage[StageName_V][FunctionType_V]["DistF"] = EamDfDistFInterpolate
                DicEamDfInterpolateValueStage[StageName_V][FunctionType_V]["DistF"] = EamDfDistFInterpolateValue

                EamDfDistPhiInterpolate = PchipFunc(EamDfGroup["Dist"], EamDfGroup["Phi"])
                # add_boundary_knots(EamDfDistPhiInterpolate, 1, 2)
                EamDfDistPhiInterpolateValues = PchipValue(EamDfGroup["Dist"], EamDfGroup["Phi"], EamDfGridDist)
                DicEamDfInterpolateStage[StageName_V][FunctionType_V]["DistPhi"] = EamDfDistPhiInterpolate
                DicEamDfInterpolateValueStage[StageName_V][FunctionType_V]["DistPhi"] = EamDfDistPhiInterpolateValues

            # </editor-fold>

            if MeasureRho_V:
                if DeepAnalysis_V:
                    # <editor-fold desc="DicEamDfStage">
                    EamDfGroupFDer = EamDfGroup["FDer"]
                    DicEamDfStage[StageName][FunctionType_V]["FDer"] = EamDfGroupFDer
                    # </editor-fold>

                    # <editor-fold desc="Interpolate Functions">
                    EamDfDistFDerInterpolate = PchipFunc(EamDfGroup["Dist"], EamDfGroup["FDer"])
                    # add_boundary_knots(EamDfDistFDerInterpolate, 1, 2)
                    EamDfDistFDerInterpolateValue = PchipValue(EamDfGroup["Dist"], EamDfGroup["FDer"], EamDfGridDist)
                    DicEamDfInterpolateStage[StageName_V][FunctionType_V]["DistFDer"] = EamDfDistFDerInterpolate
                    DicEamDfInterpolateValueStage[StageName_V][FunctionType_V]["DistFDer"] = EamDfDistFDerInterpolateValue
                    # </editor-fold>

    # </editor-fold>
    # </editor-fold>

    # <editor-fold desc="**********  Dist to Rho">
    if PrintSectionTitle_V: print("**********  Dist to Rho")
    if (MeasureRho_V) or (RunLammpsEph_V):
        if FunctionType_V == "Qsd":
            for Direction in DirectionList_V:

                # <editor-fold desc="Ungrouping">
                EamDfGroup = EamDfGrouped.get_group(Direction).reset_index(drop=True)
                # </editor-fold>

                # <editor-fold desc="EAM Dist-Rho interpolate">
                EamDfGroup = EamDfGroup.sort_values(by=["Dist"])
                EamDfInterpolateDistRho = CubicSpline(EamDfGroup["Dist"], EamDfGroup["Rho"],bc_type ="clamped")
                add_boundary_knots(EamDfInterpolateDistRho, 1, 2)
                EamDfInterpolateDistRhoValue = PchipValue(EamDfGroup["Dist"], EamDfGroup["Rho"],DicEamDfInterpolateValueStage[StageName_V][Direction]["Dist"])
                DicEamDfInterpolateStage[StageName_V][Direction]["DistRho"] = EamDfInterpolateDistRho
                DicEamDfInterpolateValueStage[StageName_V][Direction]["DistRho"] = EamDfInterpolateDistRhoValue
                # </editor-fold>

                # <editor-fold desc="RunLammps">
                EamDfDistMin = EamDf["Dist"].min()
                EamDfDistMax = EamDf["Dist"].max()
                DicRunLammpsRange[Direction]["Dist"]["Start"] = EamDfDistMin
                DicRunLammpsRange[Direction]["Dist"]["Finish"] = EamDfDistMax
                RunLammpsRhoStart = EamDfInterpolateDistRho(EamDfDistMin)
                RunLammpsRhoFinish = EamDfInterpolateDistRho(EamDfDistMax)
                DicRunLammpsRange[Direction]['Rho']["Start"] = RunLammpsRhoStart
                DicRunLammpsRange[Direction]['Rho']["Finish"] = RunLammpsRhoFinish
                # </editor-fold>

                if StageName_V != "Original":

                    # <editor-fold desc="Dft">
                    DftDfDist = DicDftDf["Original"][Direction]["Dist"]
                    DftDfEnergy = DicDftDf["Original"][Direction]["DistEnergy"]
                    DftDfRho = DftDfDist.apply(EamDfInterpolateDistRho).reset_index(name='Rho').drop('index', axis=1)
                    DicDftDfStage[StageName_V][Direction]["Rho"] = DftDfRho
                    DicDftDfStage[StageName_V][Direction]["DistEnergy"] = DftDfEnergy
                    DicDftDfStage[StageName_V][Direction]["RhoEnergy"] = DftDfEnergy

                    DftDf = pd.concat([DftDfDist, DftDfRho, DftDfEnergy], axis=1)
                    DftDf = DftDf.drop_duplicates(subset=[Variable])

                    DftDfVariableMin = DftDf[Variable].min()
                    DftDfVariableMax = DftDf[Variable].max()
                    DftDfGrid = np.mgrid[DftDfVariableMin:DftDfVariableMax:DD]

                    DftDf = DftDf.sort_values(by=[Variable])
                    DftDfVariable = DftDf[Variable]
                    DftDfEnergy = DftDf["Energy"]
                    DftDfInterpolateVariableEnergy = PchipFunc(DftDfVariable, DftDfEnergy)
                    # add_boundary_knots(DftDfInterpolateVariableEnergy, 1, 2)

                    DftDfInterpolateVariableEnergyValue = DftDfInterpolateVariableEnergy(DftDfGrid)
                    # </editor-fold>

                    # <editor-fold desc="CostRange">
                    CostRangeDist = DicCostRange[Direction]["Dist"]
                    CostRangeVariable = CostRangeDist.apply(EamDfInterpolateDistRho)
                    DicCostRange[Direction][Variable] = CostRangeVariable
                    # os.system("pause")
                    # </editor-fold>

                    # <editor-fold desc="Dic Update">
                    # if not CostMeasurement_V: #This is a good practice for one direction, but if you want to check multiple directions, it won't work!
                    DicDftDfStage[StageName_V][Direction]["Dist"] = DicDftDf["Original"][Direction]["Dist"]
                    DicDftDfStage[StageName_V][Direction]["Rho"] = DftDfVariable
                    DicDftDfStage[StageName_V][Direction]["DistEnergy"] = DicDftDf["Original"][Direction]["DistEnergy"]
                    DicDftDfStage[StageName_V][Direction]["RhoEnergy"] = DftDfEnergy

                    DicDftDfInterpolateStage[StageName_V][Direction][VariableEnergy] = DftDfInterpolateVariableEnergy
                    DicDftDfInterpolateValuesStage[StageName_V][Direction][Variable] = DftDfGrid
                    DicDftDfInterpolateValuesStage[StageName_V][Direction][VariableEnergy] = DftDfInterpolateVariableEnergyValue

                    # </editor-fold>

        elif FunctionType_V == "Eos":

            # <editor-fold desc="Ungrouping">
            EamDfGroup = EamDfGrouped.get_group(FunctionType_V).reset_index(drop=True)
            # </editor-fold>

            # <editor-fold desc="EAM Dist-Rho interpolate">
            EamDfGroup = EamDfGroup.sort_values(by=["Dist"])
            EamDfInterpolateDistRho = CubicSpline(EamDfGroup["Dist"], EamDfGroup["Rho"],bc_type ="clamped")
            add_boundary_knots(EamDfInterpolateDistRho, 1, 2)
            # plt.scatter(EamDfGroup["Dist"], EamDfGroup["Rho"])
            # plt.show()

            # plt.scatter(np.mgrid[-10:10:dr],EamDfInterpolateDistRho(np.mgrid[-10:10:dr]))
            # plt.show()

            if not CostMeasurement_V:
                DicEamDfInterpolateStage[StageName_V][FunctionType_V]["DistRho"] = EamDfInterpolateDistRho
                EamDfInterpolateDistRhoValue = PchipValue(EamDfGroup["Dist"], EamDfGroup["Rho"], DicEamDfInterpolateValueStage[StageName_V][FunctionType_V]["Dist"])
                DicEamDfInterpolateValueStage[StageName_V][FunctionType_V]["DistRho"] = EamDfInterpolateDistRhoValue

            # </editor-fold>

            # <editor-fold desc="RunLammps">
            EamDfDistMin = EamDf["Dist"].min()
            EamDfDistMax = EamDf["Dist"].max()
            DicRunLammpsRange[FunctionType_V]["Dist"]["Start"] = EamDfDistMin
            DicRunLammpsRange[FunctionType_V]["Dist"]["Finish"] = EamDfDistMax
            RunLammpsRhoStart = EamDfInterpolateDistRho(EamDfDistMin)
            RunLammpsRhoFinish = EamDfInterpolateDistRho(EamDfDistMax)
            DicRunLammpsRange[FunctionType_V]['Rho']["Start"] = RunLammpsRhoStart
            DicRunLammpsRange[FunctionType_V]['Rho']["Finish"] = RunLammpsRhoFinish
            # </editor-fold>

            if StageName_V != "Original":

                # <editor-fold desc="Dft">
                DftDfDist = DicDftDf["Original"][FunctionType_V]["Dist"]
                DftDfEnergy = DicDftDf["Original"][FunctionType_V]["DistEnergy"]

                # plt.scatter(DftDfDist,DftDfEnergy)
                # plt.show()

                DftDf = pd.concat([DftDfDist, DftDfEnergy], axis=1)

                # print(DftDfDist)
                # DftDf['Rho'] = DftDf['Dist'].apply(lambda x: EamDfInterpolateDistRho(x))
                DftDfRho = DftDfDist.apply(EamDfInterpolateDistRho).reset_index(name='Rho').drop('index', axis=1)
                # print(DftDfRho)
                # plt.scatter(DftDfRho,DftDfEnergy)
                # plt.show()

                DftDf = pd.concat([DftDfDist, DftDfRho, DftDfEnergy], axis=1)
                DftDf = DftDf.drop_duplicates(subset=[Variable])
                DftDf = DftDf[DftDf[Variable] >= 0]
                DftDf = DftDf.drop(DftDf[(DftDf[Variable] < 1e-10) & (DftDf[Variable] > 0)].index)

                DftDfVariableMin = DftDf[Variable].min()
                DftDfVariableMax = DftDf[Variable].max()
                DftDfGrid = np.mgrid[DftDfVariableMin:DftDfVariableMax:DD]

                DftDf = DftDf.sort_values(by=[Variable])
                DftDfVariable = DftDf[Variable]
                DftDfEnergy = DftDf["Energy"]


                # plt.scatter(DftDfVariable,DftDfEnergy)
                # plt.show()
                DftDfInterpolateVariableEnergy = PchipFunc(DftDfVariable, DftDfEnergy)
                # add_boundary_knots(DftDfInterpolateVariableEnergy, 1, 2)

                # if not CostMeasurement_V:
                DftDfInterpolateVariableEnergyValue = DftDfInterpolateVariableEnergy(DftDfGrid)
                # plt.scatter(DftDfGrid,DftDfInterpolateVariableEnergyValue)
                # plt.show()
                # </editor-fold>

                # <editor-fold desc="CostRange">
                CostRangeDist = DicCostRange[FunctionType_V]["Dist"]
                CostRangeVariable = CostRangeDist.apply(EamDfInterpolateDistRho)
                DicCostRange[FunctionType_V][Variable] = CostRangeVariable
                # print(DicCostRange[FunctionType_V])
                # os.system("pause")
                # </editor-fold>

                # <editor-fold desc="Dic Update">
                # if not CostMeasurement_V:
                DicDftDfStage[StageName_V][FunctionType_V]["Dist"] = DicDftDf["Original"][FunctionType_V]["Dist"]
                DicDftDfStage[StageName_V][FunctionType_V]["Rho"] = DftDfVariable
                DicDftDfStage[StageName_V][FunctionType_V]["DistEnergy"] = DicDftDf["Original"][FunctionType_V]["DistEnergy"]
                DicDftDfStage[StageName_V][FunctionType_V]["RhoEnergy"] = DftDfEnergy

                DicDftDfInterpolateStage[StageName_V][FunctionType_V][VariableEnergy] = DftDfInterpolateVariableEnergy
                DicDftDfInterpolateValuesStage[StageName_V][FunctionType_V][Variable] = DftDfGrid
                DicDftDfInterpolateValuesStage[StageName_V][FunctionType_V][VariableEnergy] = DftDfInterpolateVariableEnergyValue
                # </editor-fold>
    else:
        # if not CostMeasurement_V:
        if StageName_V != "Original":
            if FunctionType_V == "Qsd":
                for Direction in DirectionList_V:
                    # <editor-fold desc="Dft">
                    DftDfDist = DicDftDf["Original"][Direction]["Dist"]
                    DftDfDistEnergy = DicDftDf["Original"][Direction]["DistEnergy"]

                    DftDfGrid = DicDftDfInterpolateValues["Original"][Direction]["Dist"]
                    DftDfInterpolateVariableEnergy = DicDftDfInterpolate["Original"][Direction]["DistEnergy"]
                    DftDfInterpolateVariableEnergyValue = DicDftDfInterpolateValues["Original"][Direction]["DistEnergy"]
                    # </editor-fold>

                    # <editor-fold desc="DicUpdate">
                    DicDftDfStage[StageName_V][Direction]["Dist"] = DftDfDist
                    DicDftDfStage[StageName_V][Direction]["DistEnergy"] = DftDfDistEnergy

                    DicDftDfInterpolateValuesStage[StageName_V][Direction]["Dist"] = DftDfGrid
                    DicDftDfInterpolateStage[StageName_V][Direction]["DistEnergy"] = DftDfInterpolateVariableEnergy
                    DicDftDfInterpolateValuesStage[StageName_V][Direction]["DistEnergy"] = DftDfInterpolateVariableEnergyValue
                    # </editor-fold>

            elif FunctionType_V == "Eos":
                # <editor-fold desc="Dft">
                DftDfDist = DicDftDf["Original"][FunctionType_V]["Dist"]
                DftDfDistEnergy = DicDftDf["Original"][FunctionType_V]["DistEnergy"]

                DftDfGrid = DicDftDfInterpolateValues["Original"][FunctionType_V]["Dist"]
                DftDfInterpolateVariableEnergy = DicDftDfInterpolate["Original"][FunctionType_V]["DistEnergy"]
                DftDfInterpolateVariableEnergyValue = DicDftDfInterpolateValues["Original"][FunctionType_V]["DistEnergy"]
                # </editor-fold>

                # <editor-fold desc="DicUpdate">
                DicDftDfStage[StageName_V][FunctionType_V]["Dist"] = DftDfDist
                DicDftDfStage[StageName_V][FunctionType_V]["DistEnergy"] = DftDfDistEnergy

                DicDftDfInterpolateValuesStage[StageName_V][FunctionType_V]["Dist"] = DftDfGrid
                DicDftDfInterpolateStage[StageName_V][FunctionType_V]["DistEnergy"] = DftDfInterpolateVariableEnergy
                DicDftDfInterpolateValuesStage[StageName_V][FunctionType_V]["DistEnergy"] = DftDfInterpolateVariableEnergyValue
                # </editor-fold>

    # </editor-fold>
    # print(DicEamDfStage[StageName][FunctionType]["Dist"])
    # print(DicEamDfStage[StageName][FunctionType]["Rho"])
    # print(DicEamDfStage[StageName][FunctionType]["Dist"].shape)
    # print(DicEamDfStage[StageName][FunctionType]["Rho"].shape)
    # print(DicEamDfInterpolateValueStage[StageName][FunctionType]["Dist"])
    # print(DicEamDfInterpolateValueStage[StageName][FunctionType]["DistRho"])
    # print(DicEamDfInterpolateValueStage[StageName][FunctionType]["Dist"].shape)
    # print(DicEamDfInterpolateValueStage[StageName][FunctionType]["DistRho"].shape)
    # plt.scatter(DicEamDfInterpolateValueStage[StageName][FunctionType]["Dist"],DicEamDfInterpolateValueStage[StageName][FunctionType]["DistRho"],color="r")
    # plt.scatter(DicEamDfStage[StageName][FunctionType]["Dist"],DicEamDfStage[StageName][FunctionType]["Rho"],color="b")
    # plt.show()
    # os.system("pause")
    # <editor-fold desc="**********  Exporting">
    if PrintSectionTitle_V: print("**********  Exporting")
    if TimeRecord_V: start_time_L2 = time.time()
    if DeepAnalysis_V:
        # print("Here")
        # print(EamDf)
        EamDf.to_csv(WorkingDirectory + "/" + Date + "-CostFunc-" + StageName_V + "-EamNumpy-" + str(TimesCalled) + ".csv", index=False)
    if TimeRecord_V: print("____________________________________" + str(time.time() - start_time_L2),
                           " seconds"); start_time_L2 = time.time()
    # </editor-fold>

    # <editor-fold desc="**********  Analysis">
    if PrintSectionTitle_V: print("**********  Analysis")
    if TimeRecord_V: start_time_L1 = time.time()

    # <editor-fold desc="^^^^^ Interpolation-PchipValue">
    if PrintSectionTitle_V: print("^^^^^ Interpolation-PchipValue")
    # <editor-fold desc="$$$$$ Type Dependency">
    # print(EamDf)
    if FunctionType_V == "Qsd":
        for Direction in DirectionList_V:
            # <editor-fold desc="Ungrouping">
            EamDfGroup = EamDfGrouped.get_group(Direction).reset_index(drop=True)
            EamDfGroup = EamDfGroup.sort_values(by=[Variable])
            # </editor-fold>

            # <editor-fold desc="Min, Max, Grid">
            RunLammps = DicRunLammpsRange[Direction]
            EamDfGrid = DicEamDfInterpolateValueStage[StageName_V][Direction][Variable]
            EamDfGroup = EamDfGroup.drop_duplicates(subset=[Variable])
            # </editor-fold>

            # <editor-fold desc="Interpolate">
            EamDfInterpolate = PchipFunc(EamDfGroup[Variable], EamDfGroup["Energy"])
            # add_boundary_knots(EamDfInterpolate, 1, 2)
            EamDfInterpolateValue = PchipValue(EamDfGroup[Variable], EamDfGroup["Energy"], EamDfGrid)
            # </editor-fold>

            # <editor-fold desc="DicUpdate">
            # if not CostMeasurement_V:
            DicEamDfInterpolateStage[StageName_V][Direction][VariableEnergy] = EamDfInterpolate
            DicEamDfInterpolateValueStage[StageName_V][Direction][Variable] = EamDfGrid
            DicEamDfInterpolateValueStage[StageName_V][Direction][VariableEnergy] = EamDfInterpolateValue
            # </editor-fold>

    elif FunctionType_V == "Eos":
        # <editor-fold desc="Ungrouping">
        EamDfGroup = EamDfGrouped.get_group(FunctionType_V).reset_index(drop=True)
        EamDfGroup = EamDfGroup.sort_values(by=[Variable])
        EamDfGroup = EamDfGroup.drop_duplicates(subset=[Variable])
        EamDfGroup = EamDfGroup[EamDfGroup[Variable] >= 1e-10]
        # </editor-fold>

        # <editor-fold desc="Min, Max, Grid">
        EamDfVariableMin = EamDfGroup[Variable].min()
        EamDfVariableMax = EamDfGroup[Variable].max()
        EamDfGrid = np.mgrid[EamDfVariableMin:EamDfVariableMax:DD]
        # </editor-fold>

        # <editor-fold desc="Interpolate">
        EamDfInterpolate = PchipFunc(EamDfGroup[Variable], EamDfGroup["Energy"])
        # add_boundary_knots(EamDfInterpolate, 1, 2)
        EamDfInterpolateValue = PchipValue(EamDfGroup[Variable], EamDfGroup["Energy"], EamDfGrid)
        # </editor-fold>

        # <editor-fold desc="DicUpdate">
        # if not CostMeasurement_V:
        DicEamDfInterpolateStage[StageName_V][FunctionType_V][VariableEnergy] = EamDfInterpolate
        DicEamDfInterpolateValueStage[StageName_V][FunctionType_V][Variable] = EamDfGrid
        DicEamDfInterpolateValueStage[StageName_V][FunctionType_V][VariableEnergy] = EamDfInterpolateValue
        # </editor-fold>

    # </editor-fold>
    # </editor-fold>

    # <editor-fold desc="^^^^^ Finding Minima Length">
    if PrintSectionTitle_V: print("^^^^^ Finding Minima Length")
    if not CostMeasurement_V:
        # print(TypeList)
        for Type in TypeList:
            # print(Type)
            # <editor-fold desc="Min and Max">
            EamDfGroup = EamDfGrouped.get_group(Type).reset_index(drop=True)
            EamDfGroup = EamDfGroup.sort_values(by=["Dist"])
            EamDfGroupDistMin = EamDfGroup["Dist"].min()
            EamDfGroupDistMax = EamDfGroup["Dist"].max()
            EamDfGroupEqEnergy = EamDfGroup["Energy"].min()
            EamDfGroupEqDist = EamDfGroup.loc[EamDfGroup["Energy"] == EamDfGroupEqEnergy, "Dist"].iloc[0]
            if DicMeasureRho[FunctionType_V]:
                EamDfGroupEqRho = EamDfGroup.loc[EamDfGroup["Energy"] == EamDfGroupEqEnergy, "Rho"].iloc[0]
            # </editor-fold>

            # <editor-fold desc="DicRunLammpsRange">
            DicEamCritical[StageName_V][Type]["Dist"]["Min"] = EamDfGroupDistMin
            DicEamCritical[StageName_V][Type]["Dist"]["Max"] = EamDfGroupDistMax
            DicEamCritical[StageName_V][Type]["Dist"]["Eq"] = EamDfGroupEqDist
            DicEamCritical[StageName_V][Type]["Energy"]["Eq"] = EamDfGroupEqEnergy
            if DicMeasureRho[FunctionType_V]:
                DicEamCritical[StageName_V][Type]["Rho"]["Eq"] = EamDfGroupEqRho
            # print(Type)
            # print(EamDfGroupEqDist)
            # print(DicEamCriticalStage)
            # print(DicEamCritical)
            # print(EamDfGroup)
            # </editor-fold>

    # </editor-fold>

    if StageNamePure == "EgtSatZblPpm" or StageNamePure == "EgtSatZblPpmEpm":
        Plotting = False
        PlottingShow = False
        WriteWeight = False

    # <editor-fold desc="^^^^^ Plotting">
    if PrintSectionTitle_V: print("^^^^^ Plotting")
    if Plotting:
        # <editor-fold desc="$$$$$ Type Dependency">
        if FunctionType_V == "Qsd":
            for Direction in DirectionList_V:
                Title = StageName_V + "-" + Direction + "-" + str(TimesCalled)
                DicPotential = DicPotential_V  # Temp! Comment out after use
                CostRange = DicCostRange[Direction]
                EamDfGroupVariable = DicEamDfStage[StageName][Direction][Variable]
                EamDfGroupEnergy = DicEamDfStage[StageName][Direction][VariableEnergy]
                EamDfInterpolate = DicEamDfInterpolateStage[StageName_V][Direction][VariableEnergy]
                EamDfGrid = DicEamDfInterpolateValueStage[StageName_V][Direction][Variable]
                EamDfInterpolateValue = DicEamDfInterpolateValueStage[StageName_V][Direction][VariableEnergy]
                PotentialVariableGridBefore = DicPotential[StageNameBefore_V][VariableGridName]
                PotentialVariableFunctionInterpolateValuesBefore = DicPotential[StageNameBefore_V][FunctionInterpolateValuesName]
                PotentialVariableGridEgtSatZbl = DicPotential["EgtSatZbl"][VariableDist]
                PotentialVariableFunctionInterpolateValuesEgtSatZbl = DicPotential["EgtSatZbl"][FunctionInterpolateValuesName]
                EamDfGridMin = EamDfGrid.min()
                EamDfGridMax = EamDfGrid.max()

                fig, ax1 = plt.subplots(figsize=(10, 10))

                # print(EamDfGroupVariable)
                # print(EamDfGroupEnergy)
                # print("grid")
                # print(EamDfGrid)
                # EamDfInterpolateValue = PchipValue(EamDfGroupVariable, EamDfGroupEnergy, EamDfGrid) #Temp! Comment out after use
                # EamDfInterpolateValue = EamDfInterpolate(EamDfGrid) #Temp! Comment out after use
                # print("value")
                # print(EamDfInterpolateValue)

                ax1.scatter(EamDfGroupVariable, EamDfGroupEnergy, color=Colors[1], label="Eam")
                ax1.plot(EamDfGrid, EamDfInterpolateValue, color=Colors[1])  # , label="Eam Spline")

                ax2 = ax1.twinx()
                ax2.plot(PotentialVariableGridEgtSatZbl,PotentialVariableFunctionInterpolateValuesEgtSatZbl, color=Colors[4], label=FunctionName + " Function-EgtSatZbl")
                ax2.plot(PotentialVariableGridBefore, PotentialVariableFunctionInterpolateValuesBefore, color=Colors[5], label=FunctionName + " Function-Before")
                ax2.plot(XNumpy, Function(XNumpy), color=Colors[3], label=FunctionName + " Function-After")
                ax2.set_ylabel(FunctionName)
                ax2.set_xlim(DicVariable[FunctionType_V][Variable]["FunctionAxisXMin"],
                             DicVariable[FunctionType_V][Variable]["FunctionAxisXMax"])
                ax2.set_ylim(DicVariable[FunctionType_V][Variable]["FunctionAxisYMin"],
                             DicVariable[FunctionType_V][Variable]["FunctionAxisYMax"])
                ax2.set_yscale(DicVariable[FunctionType_V][Variable]["FunctionAxisScale"])

                if CostMeasurement_V:
                    ax2.scatter(FittingNumpy[:, 0], FittingNumpy[:, 1], color=Colors[3])
                    ax2.axvline(x=FittingNumpy[0, 0], linestyle="dashed", color=Colors[3], linewidth=1,
                                label="Fitting Points")  # just for the sake of legend
                    for row in range(FittingNumpy.shape[0]):
                        Value = FittingNumpy[row, 0]
                        # print(Rho)
                        ax2.axvline(x=Value, linestyle="dashed", color=Colors[3], linewidth=1)

                if StageName_V != "Original":
                    CostRangeVariable = CostRange[Variable]

                    ax1.scatter(DicDftDfStage[StageName_V][Direction][Variable],DicDftDfStage[StageName_V][Direction][VariableEnergy], color=Colors[0], label="Dft")
                    ax1.plot(DicDftDfInterpolateValuesStage[StageName_V][Direction][Variable],DicDftDfInterpolateValuesStage[StageName_V][Direction][VariableEnergy], color=Colors[0])
                    # ax1.plot(CostRangeVariable, DftInterpolate(CostRangeVariable), color=Colors[0], label="Cost Range")

                    ax1.axvline(x=EamDfGridMin, linestyle="dashed", color=Colors[1], linewidth=1, label="EAM Range")
                    ax1.axvline(x=EamDfGridMax, linestyle="dashed", color=Colors[1], linewidth=1)

                    if WriteWeight:
                        CostRangeWeight = CostRange["Weight"]
                        for i, txt in enumerate(CostRangeWeight):
                            ax1.annotate(str(round(txt, 2)), (CostRange.loc[i, Variable], CostRange.loc[i, "Energy"]))

                    EosCostMax = CostRange.max()[Variable]
                    EosCostMin = CostRange.min()[Variable]
                    ax1.axvline(x=EosCostMax, linestyle="dashed", color=Colors[0], linewidth=1, label="Cost Points")
                    ax1.axvline(x=EosCostMin, linestyle="dashed", color=Colors[0], linewidth=1)

                    for index, row in CostRange.iterrows():
                        Value = row[Variable]
                        # print(Rho)
                        ax1.axvline(x=Value, linestyle="dashed", color=Colors[0], linewidth=0.25)

                ax1.set_xlabel(Variable)
                ax1.set_ylabel("Excess Energy (Ev)")
                ax2.set_ylabel("Function to Minimize")

                plt.title(Title)
                # if (MeasureRho_V) or (RunLammpsEph_V):
                #     ax1.set_xlim(DicRunLammpsRange[Direction][Variable]["Start"], DicRunLammpsRange[Direction][Variable]["Finish"])
                # else:
                #     ax1.set_xlim(0, 4)
                # ax1.set_xlim(2, 60)
                ax1.set_xlim(DicVariable[FunctionType_V][Variable]["EamAxisXMin"],
                             DicVariable[FunctionType_V][Variable]["EamAxisXMax"])
                ax1.set_ylim(DicVariable[FunctionType_V][Variable]["EamAxisYMin"],
                             DicVariable[FunctionType_V][Variable]["EamAxisYMax"])
                ax1.set_yscale(DicVariable[FunctionType_V][Variable]["EamAxisScale"])
                # ax1.set_yscale('log')
                # ax2.set_yscale('log')
                # plt.grid()
                ax1.legend(loc="upper left")
                ax2.legend(loc="upper right")
                plt.savefig(CostFunctionDirectory + "/" + Direction + "/" + Date + "-CostFunc-" + Title)
                if PlottingShow:
                    plt.show()
                plt.close()

        elif FunctionType_V == "Eos":
            Title = StageName_V + "-" + FunctionType_V + "-" + str(TimesCalled)
            DicPotential = DicPotential_V
            CostRange = DicCostRange[FunctionType_V]
            EamDfGroupVariable = DicEamDfStage[StageName][FunctionType_V][Variable]
            EamDfGroupEnergy = DicEamDfStage[StageName][FunctionType_V][VariableEnergy]
            EamDfInterpolate = DicEamDfInterpolateStage[StageName_V][FunctionType_V][VariableEnergy]
            EamDfGrid = DicEamDfInterpolateValueStage[StageName_V][FunctionType_V][Variable]
            EamDfInterpolateValue = DicEamDfInterpolateValueStage[StageName_V][FunctionType_V][VariableEnergy]
            PotentialVariableGridBefore = DicPotential[StageNameBefore_V][VariableGridName]
            PotentialVariableFunctionInterpolateValuesBefore = DicPotential[StageNameBefore_V][FunctionInterpolateValuesName]
            PotentialVariableGridEgtSatZbl = DicPotential["EgtSatZbl"][VariableDist]
            PotentialVariableFunctionInterpolateValuesEgtSatZbl = DicPotential["EgtSatZbl"][FunctionInterpolateValuesName]
            EamDfGridMin = EamDfGrid.min()
            EamDfGridMax = EamDfGrid.max()

            fig, ax1 = plt.subplots(figsize=(10, 10))

            ax1.scatter(EamDfGroupVariable, EamDfGroupEnergy, color=Colors[1], label="Eam")
            ax1.plot(EamDfGrid, EamDfInterpolateValue, color=Colors[1])#, label="Eam Spline")

            ax2 = ax1.twinx()
            ax2.plot(XNumpy, Function(XNumpy), color=Colors[3], label=FunctionName + " Function-After")

            if CostMeasurement_V:
                ax2.scatter(FittingNumpy[:, 0], FittingNumpy[:, 1], color=Colors[3])
                ax2.axvline(x=FittingNumpy[0, 0], linestyle="dashed", color=Colors[3], linewidth=1, label="Fitting Points")  # just for the sake of legend
                for row in range(FittingNumpy.shape[0]):
                    Value = FittingNumpy[row, 0]
                    # print(Rho)
                    ax2.axvline(x=Value, linestyle="dashed", color=Colors[3], linewidth=1)

            if StageName_V != "Original":
                CostRangeVariable = CostRange[Variable]
                ax1.scatter(DicDftDfStage[StageName_V][FunctionType_V][Variable], DicDftDfStage[StageName_V][FunctionType_V][VariableEnergy], color=Colors[0], label="Dft")
                ax1.plot(DicDftDfInterpolateValuesStage[StageName_V][FunctionType_V][Variable], DicDftDfInterpolateValuesStage[StageName_V][FunctionType_V][VariableEnergy], color=Colors[0])
                # ax1.plot(CostRangeVariable, DftInterpolate(CostRangeVariable), color=Colors[0], label="Cost Range")

                ax1.axvline(x=EamDfGridMin, linestyle="dashed", color=Colors[1], linewidth=1, label="EAM Range")
                ax1.axvline(x=EamDfGridMax, linestyle="dashed", color=Colors[1], linewidth=1)

                if WriteWeight:
                    CostRangeWeight = CostRange["Weight"]
                    for i, txt in enumerate(CostRangeWeight):
                        ax1.annotate(str(round(txt, 2)), (CostRange.loc[i, Variable], CostRange.loc[i, "Energy"]))

                EosCostMax = CostRange.max()[Variable]
                EosCostMin = CostRange.min()[Variable]
                ax1.axvline(x=EosCostMax, linestyle="dashed", color=Colors[0], linewidth=1, label="Cost Points")
                ax1.axvline(x=EosCostMin, linestyle="dashed", color=Colors[0], linewidth=1)

                for index, row in CostRange.iterrows():
                    Value = row[Variable]
                    # print(Rho)
                    ax1.axvline(x=Value, linestyle="dashed", color=Colors[0], linewidth=0.25)

                ax2.plot(PotentialVariableGridBefore, PotentialVariableFunctionInterpolateValuesBefore, color=Colors[4], label=FunctionName + " Function-Before")

            ax1.set_xlabel(Variable)
            ax1.set_ylabel("Excess Energy (Ev)")
            ax2.set_ylabel("Function to Minimize")

            plt.title(Title)
            # if (MeasureRho_V) or (RunLammpsEph_V):
            #     ax1.set_xlim(DicRunLammpsRange[FunctionType_V][Variable]["Start"], DicRunLammpsRange[FunctionType_V][Variable]["Finish"])
            # else:
            #     ax1.set_xlim(0, 4)
            ax1.set_xlim(DicVariable[FunctionType_V][Variable]["EamAxisXMin"],DicVariable[FunctionType_V][Variable]["EamAxisXMax"])
            ax1.set_ylim(DicVariable[FunctionType_V][Variable]["EamAxisYMin"],DicVariable[FunctionType_V][Variable]["EamAxisYMax"])
            ax1.set_yscale(DicVariable[FunctionType_V][Variable]["EamAxisScale"])
            # ax2.set_ylim(-7, 20)
            # ax1.set_yscale('log')
            # ax2.set_yscale('log')
            # plt.grid()
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")
            plt.savefig(CostFunctionDirectory + "/" + Date + "-CostFunc-" + Title)
            if PlottingShow:
                plt.show()

        # </editor-fold>
    # </editor-fold>

    Plotting = False
    PlottingShow = False

    if TimeRecord_V: print("________________________________________________________" + str(
        time.time() - start_time_L1) + "seconds"); start_time_L1 = time.time()
    # </editor-fold>

    # <editor-fold desc="**********  Cost">
    if PrintSectionTitle_V: print("**********  Cost")
    if CostMeasurement_V:
        # <editor-fold desc="^^^^^ Deviation">
        if PrintSectionTitle_V: print("^^^^^ Deviation")
        CostNumpy = np.zeros((0, 7))
        # <editor-fold desc="$$$$$ Type Dependency">
        if FunctionType_V == "Qsd":
            for Direction in DirectionList_V:
                CostRange = DicCostRange[Direction]
                # print(CostRange)
                CostRangeVariable = CostRange[Variable]
                # print(CostRangeVariable)
                CostRangeWeight = CostRange["Weight"]
                # os.system("pause")
                # print(DicEamDfInterpolate[StageName_V])

                for Row in range(len(CostRangeVariable)):
                    Dist = float(CostRangeVariable.iloc[Row])
                    Weight = float(CostRangeWeight.iloc[Row])
                    EamEnergy = EamDfInterpolate(Dist)
                    DftEnergy = DicDftDfInterpolateStage[StageName_V][Direction][VariableEnergy](Dist)
                    # print("EamEnergy is: " + str(EamEnergy))
                    # print("DftEnergy is: " + str(DftEnergy))
                    if LogDeviation_V:
                        # print("np.log(EamEnergy) is: " + str(np.log(EamEnergy)))
                        Deviation = Weight * np.square(np.log(EamEnergy) - np.log(DftEnergy))
                    else:
                        Deviation = Weight * np.square(EamEnergy - DftEnergy)
                        # print("EamEnergy is: " + str(EamEnergy))
                    CostNew = np.array([[FunctionType_V, Direction, Row, Dist, EamEnergy, DftEnergy, Deviation]])
                    # print(CostNew)
                    CostNumpy = np.append(CostNumpy, CostNew, axis=0)
                    # print(CostNumpy)
                    # os.system("pause")

        elif FunctionType_V == "Eos":
            CostRange = DicCostRange[FunctionType_V]
            CostRangeVariable = CostRange[Variable]
            CostRangeWeight = CostRange["Weight"]

            for Row in range(len(CostRangeVariable)):
                Dist = float(CostRangeVariable.iloc[Row])
                Weight = float(CostRangeWeight.iloc[Row])
                EamEnergy = EamDfInterpolate(Dist)
                DftEnergy = DicDftDfInterpolateStage[StageName_V][FunctionType_V][VariableEnergy](Dist)

                if LogDeviation_V:
                    Deviation = Weight * np.square(np.log(EamEnergy) - np.log(DftEnergy))
                else:
                    Deviation = Weight * np.square(EamEnergy - DftEnergy)
                if math.isnan(Deviation):
                    print("Amir: Nan Error on: " + str(Dist))
                CostNew = np.array([[FunctionType_V, "Hydrostatic", Row, Dist, EamEnergy, DftEnergy, Deviation]])
                CostNumpy = np.append(CostNumpy, CostNew, axis=0)
        # </editor-fold>
        # print("CostNumpy is:\n" + str(CostNumpy))
        CostNumpyDeviation = CostNumpy[:, 6].astype(np.float64)
        # if FunctionType_V == "Eos":
        #     os.system("pause")
        # print("CostNumpyDeviation is:\n" + str(CostNumpyDeviation))
        CostNumpyDeviationSum = CostNumpyDeviation.sum()
        # print("CostNumpyDeviationSum is:\n" + str(CostNumpyDeviationSum))

        # CostDf = pd.DataFrame(data=CostNumpy, columns=["Direction","Row","Dist","EamEnergy","DftEnergy","Deviation"])
        # CostDfSum = CostDf["Deviation"].sum()
        # print("CostNumpySum is: " + str(CostDfSum))

        # </editor-fold>

        # <editor-fold desc="^^^^^ Export">
        if PrintSectionTitle_V: print("^^^^^ Export")
        if DeepAnalysis_V:
            np.savetxt(CostFunctionDirectory + "/" + Date + "-CostFunc-" + StageName_V + "-CostNumpy-" + str(TimesCalled) + ".csv",
                       CostNumpy, fmt="%s",delimiter=",",
                       header="FunctionType, Direction, Row, DistNew, EamEnergy, DftEnergy, Deviation",comments="")
        # </editor-fold>

        # <editor-fold desc="^^^^^ Deviation History">
        if PrintSectionTitle_V: print("^^^^^ Deviation History")
        CostNumpyDeviationSumNumpy = np.array([[TimesCalled, CostNumpyDeviationSum]])
        CostNumpyDeviationSumDf = pd.DataFrame(
            data={"TimesCalled": [TimesCalled], "CostNumpyDeviationSum": [CostNumpyDeviationSum]})

        CostNumpyDeviationSumDf.to_csv(
            CostFunctionDirectory + "/" + Date + "-CostFunc-" + StageName_V + "-CostNumpyDeviationSumDf.csv", mode="a",
            index=False, header=False)
        # </editor-fold>
    if TimeRecord_V: print("________________________________________________________" + str(
        time.time() - start_time_L1) + "seconds"); start_time_L1 = time.time()
    # </editor-fold>

    # <editor-fold desc="**********  Dic Update: Global">
    if PrintSectionTitle_V: print("**********  Dic Update: Global")
    if TimeRecord_V: start_time_L1 = time.time()
    if not CostMeasurement_V:
        DicEamDf = DicEamDf | DicEamDfStage
        DicEamDfInterpolate = DicEamDfInterpolate | DicEamDfInterpolateStage
        DicEamDfInterpolateValue = DicEamDfInterpolateValue | DicEamDfInterpolateValueStage

        DicDftDf = DicDftDf | DicDftDfStage
        DicDftDfInterpolate = DicDftDfInterpolate | DicDftDfInterpolateStage
        DicDftDfInterpolateValues = DicDftDfInterpolateValues | DicDftDfInterpolateValuesStage

        DicPotential = DicPotential | DicPotentialStage
        # </editor-fold>
    if TimeRecord_V: print("________________________________________________________" + str(
        time.time() - start_time_L1) + "seconds"); start_time_L1 = time.time()
    # </editor-fold>

    # <editor-fold desc="**********  Live Report">
    if PrintSectionTitle_V: print("**********  Live Report")
    if TimeRecord_V: start_time_L1 = time.time()
    if CostMeasurement_V:
        print(StageName_V + "\t" + str(FunctionType_V) + "\t" + str(TimesCalled) + "\t" + str(CostNumpyDeviationSum))
        return CostNumpyDeviationSum
    else:
        return DicEamDf, DicEamDfInterpolate, DicEamDfInterpolateValue, DicPotential, DicEamCritical, DicCostRange, DicDftDf, DicDftDfInterpolate, DicDftDfInterpolateValues
    if TimeRecord_V: print("________________________________________________________" + str(
        time.time() - start_time_L1) + "seconds"); start_time_L1 = time.time()
    # </editor-fold>

# </editor-fold>

# <editor-fold desc="CostFunctionOnTheFly">
print("CostFunctionOnTheFly")

# CorePool = mp.Pool(CoreNumber)
# print(CorePool)
# os.system("pause")

def CostFunctionOnTheFly(FittingNumpyYvalue_V,
                 FunctionType_V, DicPotential_V,
                 StageName_V, StageNameBefore_V, DirectionList_V,
                 PotentialName_V,
                 FittingNumpyXvalue_V, TransitionPoints_V,
                 DicEamDf_V, DicEamDfInterpolate_V,DicEamDfInterpolateValue_V,DicEamCriticalRead_V,
                 DicDftDf_V,DicDftDfInterpolate_V,DicDftDfInterpolateValue_V,
                 DicRunLammpsRange_V,DicCostRange_V, DicRev_V,
                 EamSource_V,  # Extract,ExtractEph,RunExtract,Read
                 ReadEamEnergyFrom_V,
                 CurrentDirectory_V, LammpsTempAddress_V,
                 DeepAnalysis_V, PrintSectionTitle_V, TimeRecord_V, LogDeviation_V, MultiProcessor_V,
                 RunLammpsOnlyCostRange_V, SameDirectory_V, MirrorAtEq_V,
                 MeasureRho_V, RunLammpsEph_V, RhoFinderName_V,MeasureDFer_V,
                 WriteLog_V, WriteDump_V, WriteReport_V,
                 CostMeasurement_V, RunCostFuncOnTheFly_V):

    # <editor-fold desc="**********  Global">
    if PrintSectionTitle_V: print("**********  Global")
    if RunCostFuncOnTheFly_V:
        global DicEamDf
        global DicEamDfInterpolate
        global DicEamDfInterpolateValue
        global DicDftDf
        global DicDftDfInterpolate
        global DicDftDfInterpolateValues
        global DicPotential
        global DicCostRange
    # </editor-fold>

    DicEamCritical = DicEamCriticalRead_V

    FunctionTypeList = ["Eos", "Qsd"]

    for FunctionType in FunctionTypeList:
        if PrintSectionTitle_V: print("^^^^ " + FunctionType)
        print("^^^^ " + FunctionType)

        # <editor-fold desc="Properties">
        if PrintSectionTitle_V: print("Properties")
        # PotentialNameNew = PotentialName + "_" + StageName_V + ".eampot"
        # print(PotentialNameNew)
        # os.system("pause")
        WorkingDirectory = CurrentDirectory + "/" + FunctionType + "/" + StageName_V + "/OnTheFly"
        WorkingDirectoryPath = Path(WorkingDirectory)
        TypeList = DicTypeList[FunctionType]
        MeasureRho = DicMeasureRho[FunctionType]
        MeasureFDer = DicMeasureFDer[FunctionType]

        if FunctionType == "Eos":
            Variable = "Rho"
            VariableDist = "RhoDist"
            VariableEnergy = "RhoEnergy"
            VariableAxis = "Rho"
            FunctionName = "F"
        elif FunctionType == "Qsd":
            Variable = "Dist"
            VariableDist = "PhysicalDist"
            VariableEnergy = "DistEnergy"
            VariableAxis = r"R " + "(" + "Å" + r")"
            FunctionName = "Phi"
        # </editor-fold>

        # <editor-fold desc="Run and Extract">
        if PrintSectionTitle_V: print("Run and Extract")
        if RunCostFuncOnTheFly_V:

            # <editor-fold desc="Delete Previous - Directory">
            # shutil.rmtree(WorkingDirectoryPath)
            # </editor-fold>

            # <editor-fold desc="CostFunction">
            if PrintSectionTitle_V: print("CostFunction")
            CostFunctionResult = CostFunction(FittingNumpyYvalue_V,
                 FunctionType, DicPotential_V,
                 StageName_V, StageNameBefore_V, TypeList,
                 PotentialName_V,
                 FittingNumpyXvalue_V, TransitionPoints_V,
                 DicEamDf_V, DicEamDfInterpolate_V,DicEamDfInterpolateValue_V, DicEamCritical,
                 DicDftDf_V,DicDftDfInterpolate_V,DicDftDfInterpolateValue_V,
                 DicRunLammpsRange_V,DicCostRange_V, DicRev_V,
                 EamSource_V,  # Extract,ExtractEph,RunExtract,Read
                 ReadEamEnergyFrom_V,
                 CurrentDirectory_V, LammpsTempAddress_V,
                 DeepAnalysis_V, PrintSectionTitle_V, TimeRecord_V, LogDeviation_V, MultiProcessor_V,
                 RunLammpsOnlyCostRange_V, SameDirectory_V, MirrorAtEq_V,
                 MeasureRho, RunLammpsEph_V, RhoFinderName_V, MeasureFDer,
                 WriteLog_V, WriteDump_V, WriteReport_V,
                 CostMeasurement_V)

            DicEamDf = CostFunctionResult[0]
            DicEamDfInterpolate = CostFunctionResult[1]
            DicEamDfInterpolateValue = CostFunctionResult[2]
            DicPotential = CostFunctionResult[3]
            DicEamCritical = CostFunctionResult[4]
            DicCostRange = CostFunctionResult[5]
            DicDftDf = CostFunctionResult[6]
            DicDftDfInterpolate = CostFunctionResult[7]
            DicDftDfInterpolateValues = CostFunctionResult[8]
            # print("HERE1")
            # print(StageName_V)
            # print(FunctionType)
            # print(DicEamCritical[StageName_V])
            # os.system("pause")
            # </editor-fold>

            # <editor-fold desc="EXTRACTING">
            if PrintSectionTitle_V: print("EXTRACTING")
            FileAddress = WorkingDirectory + "/" + Date + "-CostFunc-" + StageName_V + "-EamNumpy-0.csv"
            EamDf = pd.read_csv(FileAddress)
            # print("Size of EamDf is:")
            # print(EamDf.shape)
            # </editor-fold>

            # <editor-fold desc="Export Dictionary">
            if PrintSectionTitle_V: print("Export Dictionary")
            with open(Date + "-DicEamDf.pck", 'wb') as file_handle:
                pickle.dump(DicEamDf, file_handle)

            with open(Date + "-DicEamDfInterpolate.pck", 'wb') as file_handle:
                pickle.dump(DicEamDfInterpolate, file_handle)

            with open(Date + "-DicEamDfInterpolateValue.pck", 'wb') as file_handle:
                pickle.dump(DicEamDfInterpolateValue, file_handle)

            with open(Date + "-DicPotential.pck", 'wb') as file_handle:
                pickle.dump(DicPotential, file_handle)

            with open(Date + "-DicEamCritical.pck", 'wb') as file_handle:
                pickle.dump(DicEamCritical, file_handle)

            with open(Date + "-DicCostRange.pck", 'wb') as file_handle:
                pickle.dump(DicCostRange, file_handle)

            with open(Date + "-DicDftDf.pck", 'wb') as file_handle:
                pickle.dump(DicDftDf, file_handle)

            with open(Date + "-DicDftDfInterpolate.pck", 'wb') as file_handle:
                pickle.dump(DicDftDfInterpolate, file_handle)

            with open(Date + "-DicDftDfInterpolateValues.pck", 'wb') as file_handle:
                pickle.dump(DicDftDfInterpolateValues, file_handle)

            # </editor-fold>

        else:
            # <editor-fold desc="EXTRACTING">
            if PrintSectionTitle_V: print("EXTRACTING")
            FileAddress = WorkingDirectory + "/" + Date + "-CostFunc-" + StageName_V + "-EamNumpy-0.csv"
            EamDf = pd.read_csv(FileAddress)
            # print("Size of EamDf is:")
            # print(EamDf.shape)
            # </editor-fold>

            # <editor-fold desc="Read Dictionary">
            if PrintSectionTitle_V: print("Read Dictionary")
            with open(Date + "-DicEamDf.pck", 'rb') as file_handle:
                DicEamDf = pickle.load(file_handle)

            with open(Date + "-DicEamDfInterpolate.pck", 'rb') as file_handle:
                DicEamDfInterpolate = pickle.load(file_handle)

            with open(Date + "-DicEamDfInterpolateValue.pck", 'rb') as file_handle:
                DicEamDfInterpolateValue = pickle.load(file_handle)

            with open(Date + "-DicPotential.pck", 'rb') as file_handle:
                DicPotential = pickle.load(file_handle)

            with open(Date + "-DicEamCritical.pck", 'rb') as file_handle:
                DicEamCritical = pickle.load(file_handle)

            with open(Date + "-DicCostRange.pck", 'rb') as file_handle:
                DicCostRange = pickle.load(file_handle)

            with open(Date + "-DicDftDf.pck", 'rb') as file_handle:
                DicDftDf = pickle.load(file_handle)

            with open(Date + "-DicDftDfInterpolate.pck", 'rb') as file_handle:
                DicDftDfInterpolate = pickle.load(file_handle)

            with open(Date + "-DicDftDfInterpolateValues.pck", 'rb') as file_handle:
                DicDftDfInterpolateValues = pickle.load(file_handle)

            # </editor-fold>

        # </editor-fold>

        # <editor-fold desc="EAM">
        if PrintSectionTitle_V: print("EAM")
        Plotting = False
        # <editor-fold desc="Plotting Dist-Functions">
        # print("Plotting Dist-Functions")
        if DicMeasureRho[FunctionType]:
            if Plotting:
                LineWidth = 3
                FontSize = 20
                LineStyle = "-"
                ColorStyle = "red"
                EdgeColor = "black"
                EdgeWidth = 1
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                for Type in TypeList:
                    Title = StageName_V + "-" + Type + "-" + "DistRhoPhiF"
                    Counter = 0
                    for Stage in StageNameArray:
                        # print("Stage is : " + str(Stage))
                        # print("Type is : " + str(Type))
                        axs[0].scatter(DicEamDf[Stage][Type]["Dist"], DicEamDf[Stage][Type]["Rho"],
                                       color=Colors[Counter], edgecolors=EdgeColor, linewidth=EdgeWidth, label=Stage)
                        axs[0].plot(DicEamDfInterpolateValue[Stage][Type]["Dist"],DicEamDfInterpolateValue[Stage][Type]["DistRho"],
                                    color=Colors[Counter],linestyle=LineStyle, linewidth=LineWidth)


                        ax12 = axs[1].twinx()
                        axs[1].scatter(DicEamDf[Stage][Type]["Dist"], DicEamDf[Stage][Type]["F"], color=Colors[Counter],
                                       edgecolors=EdgeColor, linewidth=EdgeWidth, label=Stage)
                        axs[1].plot(DicEamDfInterpolateValue[Stage][Type]["Dist"], DicEamDfInterpolateValue[Stage][Type]["DistF"],
                                    color=Colors[Counter], linestyle=LineStyle, linewidth=LineWidth)

                        # ax12.scatter(DicEamDf[Stage][Type]["Dist"], DicEamDf[Stage][Type]["FDer"],
                        #              color=Colors[Counter], marker="x", edgecolors=EdgeColor, linewidth=EdgeWidth, label=Stage)
                        # ax12.plot(DicEamDfInterpolateValue[Stage][Type]["Dist"], DicEamDfInterpolateValue[Stage][Type]["DistFDer"],
                        #           color=Colors[Counter], linestyle="--", linewidth=LineWidth)


                        axs[2].scatter(DicEamDf[Stage][Type]["Dist"], DicEamDf[Stage][Type]["Phi"],
                                       color=Colors[Counter], edgecolors=EdgeColor, linewidth=EdgeWidth, label=Stage)
                        axs[2].plot(DicEamDfInterpolateValue[Stage][Type]["Dist"], DicEamDfInterpolateValue[Stage][Type]["DistPhi"],
                                    color=Colors[Counter],linestyle=LineStyle, linewidth=LineWidth)

                        Counter += 1

                    axs[0].set_title("(a)", fontsize=FontSize)
                    axs[0].set_xlabel(r"R " + "(" + "Å" + r")")
                    axs[0].set_ylabel(r"ρ (" + "1/Å$^³$)")
                    # axs[0].set_xlim(1, 8)
                    axs[0].set_ylim(-10, 100)
                    # axs[0].set_xscale("log")
                    # axs[0].set_yscale("log")
                    axs[0].spines[["right", "top"]].set_visible(False)
                    # axs[0].legend(prop={"size": 12})

                    axs[1].set_title("(b)", fontsize=FontSize)
                    axs[1].set_xlabel(r"R " + "(" + "Å" + r")")
                    axs[1].set_ylabel("F (eV)")
                    # axs[1].set_xlim(-0.5, 5)
                    axs[1].set_ylim(-20, 100)
                    # axs[1].set_xscale("log")
                    # axs[1].set_yscale("log")
                    axs[1].spines[["right", "top"]].set_visible(False)
                    # axs[1].legend(prop={"size": 12})
                    ax12.set_ylabel("dF/dr (eV/Å)")

                    axs[2].set_title("(c)", fontsize=FontSize)
                    axs[2].set_xlabel(r"R " + "(" + "Å" + r")")
                    axs[2].set_ylabel("φ (eV)")
                    # axs[2].set_xlim(-0.5, 5)
                    axs[2].set_ylim(0, 50)
                    # axs[2].set_xscale("log")
                    # axs[2].set_yscale("log")
                    axs[2].spines[["right", "top"]].set_visible(False)
                    axs[2].legend(prop={"size": 12})

                    for ax in axs:
                        ax.spines['top'].set_visible(True)
                        ax.spines['right'].set_visible(True)
                        ax.spines['bottom'].set_visible(True)
                        ax.spines['left'].set_visible(True)

                    plt.tight_layout()
                    fig.suptitle(Title, verticalalignment="baseline")
                    fig.savefig(Date + "-" + Title + ".png")
                    if PlottingShow:
                        plt.show()

        # </editor-fold>
        Plotting = False

        Plotting = False
        # <editor-fold desc="Plotting Dist-Energy">
        # print("Plotting Dist-Energy")
        if Plotting:
            Title = StageName_V + "-" + FunctionType + "-" + "DistEnergy"
            LineWidth = 3
            FontSize = 20
            LineStyle = "-"
            ColorStyle = "red"
            EdgeColor = "black"
            EdgeWidth = 1
            Rows = 1
            Columns = len(TypeList)
            fig, axs = plt.subplots(Rows, Columns, squeeze=False, figsize=(5 * Columns, 5))
            TypeCount = 0
            for Type in TypeList:
                StageCount = 0
                for Stage in StageNameArray:
                    # print("Type is : " + str(Type))
                    # print("TypeCount is : " + str(TypeCount))
                    # print("Stage is : " + str(Stage))
                    # print("StageCount is : " + str(StageCount))
                    # print(DicEamDf[Stage][Type]["Dist"])
                    axs[0][TypeCount].scatter(DicEamDf[Stage][Type]["Dist"], DicEamDf[Stage][Type]["DistEnergy"],
                                              color=Colors[StageCount],edgecolors=EdgeColor,linewidth=EdgeWidth, label=Stage)
                    axs[0][TypeCount].plot(DicEamDfInterpolateValue[Stage][Type]["Dist"], DicEamDfInterpolateValue[Stage][Type]["DistEnergy"],
                                           color=Colors[StageCount], linestyle=LineStyle, linewidth=LineWidth)
                    StageCount += 1
                axs[0][TypeCount].set_title(Type, fontsize=FontSize)
                axs[0][TypeCount].set_xlabel(r"R " + "(" + "Å" + r")")
                axs[0][TypeCount].set_ylabel("Excess Energy (Ev)")
                # axs[0].set_xlim(1, 8)
                # axs[0].set_ylim(-0.1, 1)
                # axs[0].set_xscale("log")
                # axs[0].set_yscale("log")
                axs[0][TypeCount].spines[["right", "top"]].set_visible(False)
                # axs[0].legend(prop={"size": 12})
                TypeCount += 1
            axs[0][Columns - 1].legend(prop={"size": 12})
            plt.tight_layout()
            fig.savefig(Date + "-" + Title + ".png")
            if PlottingShow:
                plt.show()
        # </editor-fold>
        Plotting = False

        Plotting = False
        # <editor-fold desc="Plotting Rho-Energy">
        # print("Plotting Rho vs. Energy")
        if DicMeasureRho[FunctionType]:
            if Plotting:
                Title = StageName_V + "-" + FunctionType + "-" + "RhoEnergy"
                LineWidth = 3
                FontSize = 20
                LineStyle = "-"
                ColorStyle = "red"
                EdgeColor = "black"
                EdgeWidth = 1
                Rows = 1
                Columns = len(TypeList)
                fig, axs = plt.subplots(Rows, Columns, squeeze=False, figsize=(5 * Columns, 5))
                TypeCount = 0
                for Type in TypeList:
                    StageCount = 0
                    for Stage in StageNameArray:
                        # print("Type is : " + str(Type))
                        # print("TypeCount is : " + str(TypeCount))
                        # print("Stage is : " + str(Stage))
                        # print("StageCount is : " + str(StageCount))
                        # print(DicEamDf[Stage][Type]["Rho"])
                        axs[0][TypeCount].scatter(DicEamDf[Stage][Type]["Rho"], DicEamDf[Stage][Type]["RhoEnergy"],
                                                  color=Colors[StageCount], edgecolors=EdgeColor, linewidth=EdgeWidth,
                                                  label=Stage)
                        axs[0][TypeCount].plot(DicEamDfInterpolateValue[Stage][Type]["Rho"],
                                               DicEamDfInterpolateValue[Stage][Type]["RhoEnergy"],
                                               color=Colors[StageCount], linestyle=LineStyle, linewidth=LineWidth)
                        StageCount += 1
                    axs[0][TypeCount].set_title(Type, fontsize=FontSize)
                    axs[0][TypeCount].set_xlabel("Rho")
                    axs[0][TypeCount].set_ylabel("Excess Energy (Ev)")
                    # axs[0].set_xlim(1, 8)
                    # axs[0].set_ylim(-0.1, 1)
                    # axs[0].set_xscale("log")
                    # axs[0].set_yscale("log")
                    axs[0][TypeCount].spines[["right", "top"]].set_visible(False)
                    # axs[0].legend(prop={"size": 12})
                    TypeCount += 1
                axs[0][Columns - 1].legend(prop={"size": 12})
                plt.tight_layout()
                fig.savefig(Date + "-" + Title + ".png")
                if PlottingShow:
                    plt.show()

        # </editor-fold>
        Plotting = False

        # </editor-fold>


        Plotting = True
        PlottingShow = False
        # <editor-fold desc="Dft-All">
        if PrintSectionTitle_V: print("Dft-All")
        if Plotting:
            if StageName != "Original":
            # if StageNamePure == "EgtSatZblPpmEpm" and FunctionType == "Eos":
            # if StageNamePure == "EgtSatZblPpmEpm":
            # if StageNamePure == "EgtSatZblPpm" or StageNamePure == "EgtSatZblPpmEpm":
            # if FunctionType == "Eos":
                # print("StageName is : " + str(StageName))
                Title = StageName_V + "-" + FunctionType + "-" + "DftRhoEnergy"
                LineWidth = 3
                FontSize = 20
                LineStyle = "-"
                ColorStyle = "red"
                EdgeColor = "black"
                EdgeWidth = 1
                Rows = 1
                Columns = len(TypeList)
                fig, axs = plt.subplots(Rows, Columns, squeeze=False, figsize=(8 * Columns, 8))
                TypeCount = 0
                for Type in TypeList:
                    # print("Type is : " + str(Type))
                    # print("TypeCount is : " + str(TypeCount))
                    # print(DicDftDf[StageName][Type])

                    # axs[0][TypeCount].scatter(DicDftDf[StageName_V][Type][Variable], DicDftDf[StageName_V][Type][VariableEnergy],
                    #                           color=Colors[3], edgecolors=EdgeColor, linewidth=EdgeWidth,
                    #                           label=FunctionType + "-" + StageName_V + "-" + Type + "-" + Variable + "-" + "DFT")
                    axs[0][TypeCount].plot(DicDftDfInterpolateValues[StageName_V][Type][Variable], DicDftDfInterpolateValues[StageName][Type][VariableEnergy],
                                           color=Colors[3], linestyle=LineStyle, linewidth=LineWidth,label=FunctionType + "-" + StageName_V + "-" + Type + "-" + Variable + "-" + "DFT")

                    # axs[0][TypeCount].scatter(DicEamDf["Egt"][Type][Variable], DicEamDf["Egt"][Type][VariableEnergy],
                    #                           color=Colors[2], edgecolors=EdgeColor, linewidth=EdgeWidth,
                    #                           label=FunctionType + "-" + "Egt" + "-" + Type + "-" + Variable + "-" + "EAM")
                    axs[0][TypeCount].plot(DicEamDfInterpolateValue["Egt"][Type][Variable], DicEamDfInterpolateValue["Egt"][Type][VariableEnergy],
                                           color=Colors[2], linestyle=LineStyle, linewidth=LineWidth,label=FunctionType + "-" + "Egt" + "-" + Type + "-" + Variable + "-" + "EAM")

                    axs[0][TypeCount].scatter(DicEamDf[StageNameBefore_V][Type][Variable], DicEamDf[StageNameBefore_V][Type][VariableEnergy],
                                              color=Colors[1], edgecolors=EdgeColor, linewidth=EdgeWidth,marker = "x",
                                              label=FunctionType + "-" + StageNameBefore_V + "-" + Type + "-" + Variable + "-" + "EAM")
                    axs[0][TypeCount].plot(DicEamDfInterpolateValue[StageNameBefore_V][Type][Variable], DicEamDfInterpolateValue[StageNameBefore_V][Type][VariableEnergy],
                                           color=Colors[1], linestyle=LineStyle, linewidth=LineWidth)

                    axs[0][TypeCount].scatter(DicEamDf[StageName_V][Type][Variable], DicEamDf[StageName_V][Type][VariableEnergy],
                                              color=Colors[0], edgecolors=EdgeColor, linewidth=EdgeWidth,marker = "o",
                                              label=FunctionType + "-" + StageName_V + "-" + Type + "-" + Variable + "-" + "EAM")
                    axs[0][TypeCount].plot(DicEamDfInterpolateValue[StageName_V][Type][Variable], DicEamDfInterpolateValue[StageName_V][Type][VariableEnergy],
                                           color=Colors[0], linestyle=LineStyle, linewidth=LineWidth)

                    if (StageNamePure == "EgtSatZblPpm") or (StageNamePure == "EgtSatZblPpmEpm"):

                        # <editor-fold desc="CostRange">
                        CostRange = DicCostRange[Type]
                        # print(CostRange)
                        for index, row in CostRange.iterrows():
                            VariableColumn = row[Variable]
                            axs[0][TypeCount].axvline(x=VariableColumn, linestyle="dashed", color=Colors[0], linewidth=0.25)
                        CostRangeVariableMax = CostRange.max()[Variable]
                        CostRangeVariableMin = CostRange.min()[Variable]
                        axs[0][TypeCount].axvline(x=CostRangeVariableMax, linestyle="dashed", color=Colors[0], linewidth=1,
                                    label="Cost Range")
                        axs[0][TypeCount].axvline(x=CostRangeVariableMin, linestyle="dashed", color=Colors[0], linewidth=1)
                        for i, txt in enumerate(CostRange["Weight"]):
                            axs[0][TypeCount].annotate(str(round(txt, 1)), (CostRange.loc[i, Variable], CostRange.loc[i, "Energy"]))
                        # </editor-fold>

                        # <editor-fold desc="RunLammpsRange">
                        RunLammpsRhoStart = DicRunLammpsRange[Type][Variable]["Start"]
                        RunLammpsRhoFinish = DicRunLammpsRange[Type][Variable]["Finish"]
                        axs[0][TypeCount].axvline(x=RunLammpsRhoStart, linestyle="dashed", color=Colors[3], linewidth=1, label="EAM measurement Range")
                        axs[0][TypeCount].axvline(x=RunLammpsRhoFinish, linestyle="dashed", color=Colors[3], linewidth=1)
                        # </editor-fold>

                        # <editor-fold desc="Function">
                        ax2 = axs[0][TypeCount].twinx()
                        # ax2.plot(FittingNumpyDist, FittingNumpyInterpolateValue, color=Colors[3])
                        # ax2.scatter(FittingNumpy[:, 0], FittingNumpy[:, 1], color=Colors[3], label=Variable + "-" + FunctionName + ": FittingNumpy")
                        # print(FittingNumpy)
                        ax2.scatter(DicPotential["EgtSatZbl"][VariableDist], DicPotential["EgtSatZbl"][FunctionName], color=Colors[4], label=Variable + "-" + FunctionName + ": Original")
                        ax2.scatter(DicPotential[StageNameBefore_V][VariableDist], DicPotential[StageNameBefore_V][FunctionName], color=Colors[5], label=Variable + "-" + FunctionName + ": " + StageNameBefore_V)
                        ax2.scatter(DicPotential[StageName_V][VariableDist], DicPotential[StageName_V][FunctionName], color=Colors[6], label=Variable + "-" + FunctionName + ": " + StageName_V)
                        # ax2.plot(RhoDist, RhoFOriginalInterpolateValues, color=Colors[4])

                        # <editor-fold desc="Fitting Points">
                        # print(FittingNumpyXvalue_V)
                        for row in range(FittingNumpyXvalue_V.shape[0]):
                            FittingPoint = FittingNumpyXvalue_V[row]
                            ax2.axvline(x=FittingPoint, linestyle="dashed", color=Colors[3], linewidth=1)
                        ax2.axvline(x=FittingPoint, linestyle="dashed", color=Colors[3], linewidth=1, label="Fitting Points")
                        # </editor-fold>

                        ax2.legend(loc="upper right")
                        ax2.set_ylabel(FunctionName)
                        ax2.set_xlim(DicVariable[FunctionType][Variable]["FunctionAxisXMin"],
                                     DicVariable[FunctionType][Variable]["FunctionAxisXMax"])
                        ax2.set_ylim(DicVariable[FunctionType][Variable]["FunctionAxisYMin"],
                                     DicVariable[FunctionType][Variable]["FunctionAxisYMax"])
                        ax2.set_yscale(DicVariable[FunctionType][Variable]["FunctionAxisScale"])
                        # </editor-fold>

                    axs[0][TypeCount].set_title(Type, fontsize=FontSize)
                    axs[0][TypeCount].set_xlabel(Variable)
                    axs[0][TypeCount].set_ylabel("Excess Energy (Ev)")
                    axs[0][TypeCount].set_xlim(DicVariable[FunctionType][Variable]["EamAxisXMin"],
                                               DicVariable[FunctionType][Variable]["EamAxisXMax"])
                    axs[0][TypeCount].set_ylim(DicVariable[FunctionType][Variable]["EamAxisYMin"],
                                               DicVariable[FunctionType][Variable]["EamAxisYMax"])
                    # axs[0].set_xscale("log")
                    # print(DicVariable[FunctionType][Variable])
                    axs[0][TypeCount].set_yscale(DicVariable[FunctionType][Variable]["EamAxisScale"])
                    axs[0][TypeCount].spines[["right", "top"]].set_visible(False)
                    # axs[0].legend(prop={"size": 12})
                    # plt.grid()
                    axs[0][TypeCount].legend(loc="upper left")

                    TypeCount += 1
                axs[0][Columns - 1].legend(prop={"size": 12},loc="upper left")
                plt.tight_layout()
                fig.savefig(Date + "-" + Title + ".png")
                if PlottingShow:
                    plt.show()
                # </editor-fold>
        # </editor-fold>
        Plotting = False
        PlottingShow = False


        Plotting = False
        PlottingShow = False
        # <editor-fold desc="Dft-Presentation">
        if PrintSectionTitle_V: print("Dft-Presentation")
        if Plotting:
            if StageName != "Original":
                # if StageNamePure == "EgtSatZblPpm" or StageNamePure == "EgtSatZblPpmEpm":
                # if FunctionType == "Eos":
                # print("StageName is : " + str(StageName))
                Title = StageName_V + "-" + FunctionType + "-" + "DftRhoEnergy"
                LineWidth = 3
                FontSize = 20
                LegendFontSize = 10
                LineStyle = "-"
                ColorStyle = "red"
                EdgeColor = "black"
                EdgeWidth = 1
                Rows = 1
                Columns = len(TypeList)
                fig, axs = plt.subplots(Rows, Columns, squeeze=False, figsize=(8 * Columns, 8))
                TypeCount = 0
                for Type in TypeList:
                    # print("Type is : " + str(Type))
                    # print("TypeCount is : " + str(TypeCount))
                    # print(DicDftDf[StageName][Type])

                    EamAfterPlot = axs[0][TypeCount].scatter(DicEamDf[StageName][Type][Variable],
                                              DicEamDf[StageName][Type][VariableEnergy],
                                              color=Colors[0], edgecolors=EdgeColor, linewidth=EdgeWidth,
                                              label="EAM After Optimization")
                    axs[0][TypeCount].plot(DicEamDfInterpolateValue[StageName][Type][Variable],
                                           DicEamDfInterpolateValue[StageName][Type][VariableEnergy],
                                           color=Colors[0], linestyle=LineStyle, linewidth=LineWidth)

                    EamBeforePlot = axs[0][TypeCount].scatter(DicEamDf["Egt"][Type][Variable],
                                              DicEamDf["Egt"][Type][VariableEnergy],
                                              color=Colors[1], edgecolors=EdgeColor, linewidth=EdgeWidth,
                                              label="EAM Before Optimization")
                    axs[0][TypeCount].plot(DicEamDfInterpolateValue["Egt"][Type][Variable],
                                           DicEamDfInterpolateValue["Egt"][Type][VariableEnergy],
                                           color=Colors[1], linestyle=LineStyle, linewidth=LineWidth)

                    print(Variable)
                    # print(DicDftDf[StageName])
                    print(DicDftDf[StageName][Type])
                    # print(DicDftDf[StageName][Type][Variable])
                    # print(DicDftDf[StageName][Type][VariableEnergy])
                    # os.system("pause")

                    DftPlot = axs[0][TypeCount].scatter(DicDftDf[StageName][Type][Variable],
                                              DicDftDf[StageName][Type][VariableEnergy],
                                              color=Colors[2], edgecolors=EdgeColor, linewidth=EdgeWidth,
                                              label="DFT")
                    axs[0][TypeCount].plot(DicDftDfInterpolateValues[StageName][Type][Variable],
                                           DicDftDfInterpolateValues[StageName][Type][VariableEnergy],
                                           color=Colors[2], linestyle=LineStyle, linewidth=LineWidth)

                    if (StageNamePure == "EgtSatZblPpm") or (StageNamePure == "EgtSatZblPpmEpm"):

                        # <editor-fold desc="CostRange">
                        # CostRange = DicCostRange[Type]
                        # # print(CostRange)
                        # for index, row in CostRange.iterrows():
                        #     VariableColumn = row[Variable]
                        #     axs[0][TypeCount].axvline(x=VariableColumn, linestyle="dashed", color=Colors[0],
                        #                               linewidth=0.25)
                        # CostRangeVariableMax = CostRange.max()[Variable]
                        # CostRangeVariableMin = CostRange.min()[Variable]
                        # axs[0][TypeCount].axvline(x=CostRangeVariableMax, linestyle="dashed", color=Colors[0],
                        #                           linewidth=1,
                        #                           label="Cost Range")
                        # axs[0][TypeCount].axvline(x=CostRangeVariableMin, linestyle="dashed", color=Colors[0],
                        #                           linewidth=1)
                        # for i, txt in enumerate(CostRange["Weight"]):
                        #     axs[0][TypeCount].annotate(str(round(txt, 1)),
                        #                                (CostRange.loc[i, Variable], CostRange.loc[i, "Energy"]))
                        # </editor-fold>

                        # <editor-fold desc="RunLammpsRange">
                        # RunLammpsRhoStart = DicRunLammpsRange[Type][Variable]["Start"]
                        # RunLammpsRhoFinish = DicRunLammpsRange[Type][Variable]["Finish"]
                        # axs[0][TypeCount].axvline(x=RunLammpsRhoStart, linestyle="dashed", color=Colors[2],
                        #                           linewidth=1, label="EAM measurement Range")
                        # axs[0][TypeCount].axvline(x=RunLammpsRhoFinish, linestyle="dashed", color=Colors[2],
                        #                           linewidth=1)
                        # </editor-fold>

                        # <editor-fold desc="Function">
                        ax2 = axs[0][TypeCount].twinx()
                        # ax2.plot(FittingNumpyDist, FittingNumpyInterpolateValue, color=Colors[3])
                        # ax2.scatter(FittingNumpy[:, 0], FittingNumpy[:, 1], color=Colors[3], label=Variable + "-" + FunctionName + ": FittingNumpy")
                        # print(FittingNumpy)
                        FuncOriginalPlot = ax2.scatter(DicPotential["EgtSatZbl"][VariableDist],
                                    DicPotential["EgtSatZbl"][FunctionName], color=Colors[4],
                                    label="Function: Original")
                        FuncBeforePlot = ax2.scatter(DicPotential[StageNameBefore_V][VariableDist],
                                    DicPotential[StageNameBefore_V][FunctionName], color=Colors[5],
                                    label="Function: Before Optimization")
                        FuncAfterPlot = ax2.scatter(DicPotential[StageName_V][VariableDist],
                                    DicPotential[StageName_V][FunctionName], color=Colors[6],
                                    label="Function: After Optimization")
                        # ax2.plot(RhoDist, RhoFOriginalInterpolateValues, color=Colors[4])

                        # <editor-fold desc="Fitting Points">
                        # for row in range(FittingNumpyXvalue_V.shape[0]):
                        #     FittingPoint = FittingNumpyXvalue_V[row]
                        #     ax2.axvline(x=FittingPoint, linestyle="dashed", color=Colors[3], linewidth=1)
                        # ax2.axvline(x=FittingPoint, linestyle="dashed", color=Colors[3], linewidth=1,
                        #             label="Fitting Points")
                        # </editor-fold>

                        # ax2.legend(prop={"size": LegendFontSize},facecolor='white', loc="upper right").set_zorder(1000)
                        ax2.set_ylabel(FunctionName)
                        ax2.set_xlim(DicVariable[FunctionType][Variable]["FunctionAxisXMin"],
                                     DicVariable[FunctionType][Variable]["FunctionAxisXMax"])
                        ax2.set_ylim(DicVariable[FunctionType][Variable]["FunctionAxisYMin"],
                                     DicVariable[FunctionType][Variable]["FunctionAxisYMax"])
                        ax2.set_yscale(DicVariable[FunctionType][Variable]["FunctionAxisScale"])
                        # </editor-fold>

                    axs[0][TypeCount].set_title(Type, fontsize=FontSize)
                    axs[0][TypeCount].set_xlabel(Variable)
                    axs[0][TypeCount].set_ylabel("Excess Energy (Ev)")
                    axs[0][TypeCount].set_xlim(DicVariable[FunctionType][Variable]["EamAxisXMin"],
                                               DicVariable[FunctionType][Variable]["EamAxisXMax"])
                    axs[0][TypeCount].set_ylim(DicVariable[FunctionType][Variable]["EamAxisYMin"],
                                               DicVariable[FunctionType][Variable]["EamAxisYMax"])
                    # axs[0].set_xscale("log")
                    # print(DicVariable[FunctionType][Variable])
                    axs[0][TypeCount].set_yscale(DicVariable[FunctionType][Variable]["EamAxisScale"])
                    axs[0][TypeCount].spines[["right", "top"]].set_visible(False)
                    # axs[0].legend(prop={"size": 12})
                    # plt.grid()
                    h, l = np.hstack([axis.get_legend_handles_labels()
                                      for axis in axs[0][TypeCount].figure.axes
                                      if axis.bbox.bounds == axs[0][TypeCount].bbox.bounds]).tolist()
                    ax2.legend(handles=h, labels=l, loc='upper right')
                    # axs[0][TypeCount].legend(prop={"size": LegendFontSize},facecolor='white', loc="upper left").set_zorder(1000)

                    TypeCount += 1
                # axs[0][Columns - 1].legend(prop={"size": LegendFontSize},facecolor='white', loc="upper left").set_zorder(1000)
                plt.tight_layout()
                fig.savefig(Date + "-" + Title + "-Presentation.png")
                if PlottingShow:
                    plt.show()
                # </editor-fold>
        # </editor-fold>
        Plotting = False
        PlottingShow = False

        Plotting = True
        PlottingShow = False
        # <editor-fold desc="EamDft-1x1">
        if PrintSectionTitle_V: print("EamDft-1x1")
        if Plotting:
            # if StageNamePure == "EgtSatZblPpm" or StageNamePure == "EgtSatZblPpmEpm":
            if StageName != "Original":
            # if StageName == "EgtSatZblPpmEpm-4" or StageName == "EgtSatZblPpm-4":
                # print("StageName is : " + str(StageName))
                LineWidth = 3
                FontSize = 20
                LegendFontSize = 10
                LineStyle = "-"
                ColorStyle = "red"
                EdgeColor = None#"black"
                MarkerSize = 40
                EdgeWidth = 1
                Rows = 1
                Columns = 1#len(TypeList)

                TypeCount = 0
                for Type in TypeList:
                    # print(Type)
                    # print(DicDftDf[StageName][Type])
                    Title = PotentialName + "-" + StageName_V + "-" + FunctionType + "-" + "EamDft1x1" + "-" + Type
                    # if (StageNamePure == "EgtSatZblPpm") or (StageNamePure == "EgtSatZblPpmEpm"):
                    fig, axs = plt.subplots(figsize=(4.2, 4))

                    axs.plot(DicDftDfInterpolateValues[StageName][Type][Variable],
                                           DicDftDfInterpolateValues[StageName][Type][VariableEnergy],
                                           color=Colors[4], linestyle=LineStyle, linewidth=LineWidth,zorder=3)
                    DftPlot = axs.scatter(DicDftDf[StageName][Type][Variable],
                                              DicDftDf[StageName][Type][VariableEnergy],
                                              color=Colors[4], edgecolors=EdgeColor, linewidth=EdgeWidth, s=MarkerSize,
                                              label="DFT",zorder=4)

                    axs.plot(DicEamDfInterpolateValue[StageName][Type][Variable],
                                           DicEamDfInterpolateValue[StageName][Type][VariableEnergy],
                                           color=Colors[0], linestyle=LineStyle, linewidth=LineWidth,zorder=5)
                    EamAfterPlot = axs.scatter(DicEamDf[StageName][Type][Variable],
                                              DicEamDf[StageName][Type][VariableEnergy],
                                              color=Colors[0], edgecolors=EdgeColor, linewidth=EdgeWidth, s=MarkerSize,
                                              label="EAM: After",zorder=6)


                    axs.plot(DicEamDfInterpolateValue["Egt"][Type][Variable],
                                           DicEamDfInterpolateValue["Egt"][Type][VariableEnergy],
                                           color=Colors[1], linestyle=LineStyle, linewidth=LineWidth,zorder=7)
                    EamBeforePlot = axs.scatter(DicEamDf["Egt"][Type][Variable],
                                              DicEamDf["Egt"][Type][VariableEnergy],
                                              color=Colors[1], edgecolors=EdgeColor, linewidth=EdgeWidth, s=MarkerSize,
                                              label="EAM: Before",zorder=8)

                    if Type == "Eos":
                        axs.set_xlabel('Rho')
                        axs.set_ylabel("Excess Energy (Ev)")
                        axs.set_xlim(0.9,7.1)
                        axs.set_ylim(-0.5,20)
                        axs.set_yscale("linear")
                        # axs.spines[["right", "top"]].set_visible(True)
                        h, l = np.hstack([axis.get_legend_handles_labels()
                                          for axis in axs.figure.axes
                                          if axis.bbox.bounds == axs.bbox.bounds]).tolist()
                        plt.legend(handles=h, labels=l, loc='upper left', frameon=False)
                    elif Type == "A":
                        axs.set_xlabel('Distance (Å)')
                        axs.set_ylabel("Excess Energy (Ev)")
                        axs.set_xlim(0.48, 2.5)
                        axs.set_ylim(2E-2, 50)
                        axs.set_yscale("log")
                        # axs.spines[["right", "top"]].set_visible(True)
                    elif Type == "C":
                        axs.set_xlabel('Distance (Å)')
                        axs.set_ylabel("Excess Energy (Ev)")
                        axs.set_xlim(0.2, 4.25)
                        axs.set_ylim(5E-2, 60)
                        axs.set_yscale("log")
                        # axs.spines[["right", "top"]].set_visible(True)
                    elif Type == "Ac":
                        axs.set_xlabel('Distance (Å)')
                        axs.set_ylabel("Excess Energy (Ev)")
                        axs.set_xlim(0.49, 2.6)
                        axs.set_ylim(2E-2, 40)
                        axs.set_yscale("log")
                        # axs.spines[["right", "top"]].set_visible(True)

                    TypeCount += 0#1
                    plt.tight_layout()
                    plt.savefig(Date + "-" + Title + ".png")
                    if PlottingShow:
                        plt.show()
                    plt.close()
                # </editor-fold>
        # </editor-fold>
        Plotting = False
        PlottingShow = False

        Plotting = False
        PlottingShow = True
        # <editor-fold desc="EamDftFunction-1x1">
        if PrintSectionTitle_V: print("EamDftFunction-1x1")
        if Plotting:
            # if StageName != "Original":
            if StageName == "EgtSatZblPpmEpm-4" or StageName == "EgtSatZblPpm-4":
                # print("StageName is : " + str(StageName))
                Title = StageName_V + "-" + FunctionType + "-" + "DftRhoEnergy"
                LineWidth = 3
                FontSize = 20
                LegendFontSize = 10
                LineStyle = "-"
                ColorStyle = "red"
                EdgeColor = None#"black"
                MarkerSize = 40
                EdgeWidth = 1
                Rows = 1
                Columns = 1#len(TypeList)

                TypeCount = 0
                for Type in TypeList:

                    if (StageNamePure == "EgtSatZblPpm") or (StageNamePure == "EgtSatZblPpmEpm"):
                        fig, axs = plt.subplots(figsize=(4.5, 4))

                        # <editor-fold desc="Function">
                        ax2 = axs.twinx()
                        axs.set_zorder(1)  # default zorder is 0 for ax1 and ax2
                        axs.patch.set_visible(False)  # prevents ax1 from hiding ax2
                        # ax2.plot(FittingNumpyDist, FittingNumpyInterpolateValue, color=Colors[3])
                        # ax2.scatter(FittingNumpy[:, 0], FittingNumpy[:, 1], color=Colors[3], label=Variable + "-" + FunctionName + ": FittingNumpy")
                        # print(FittingNumpy)
                        FuncOriginalPlot = ax2.plot(DicPotential["Egt"][VariableDist],
                                    DicPotential["Egt"][FunctionName], color=Colors[4],
                                    label=DicVariable[FunctionType][Variable]["FunctionName"] + ": Before",zorder=1)
                        FuncAfterPlot = ax2.plot(DicPotential[StageName_V][VariableDist],
                                    DicPotential[StageName_V][FunctionName], color=Colors[6],
                                    label=DicVariable[FunctionType][Variable]["FunctionName"] + ": After",zorder=2)
                        # ax2.plot(RhoDist, RhoFOriginalInterpolateValues, color=Colors[4])

                        # <editor-fold desc="Fitting Points">
                        # for row in range(FittingNumpyXvalue_V.shape[0]):
                        #     FittingPoint = FittingNumpyXvalue_V[row]
                        #     ax2.axvline(x=FittingPoint, linestyle="dashed", color=Colors[3], linewidth=1)
                        # ax2.axvline(x=FittingPoint, linestyle="dashed", color=Colors[3], linewidth=1,
                        #             label="Fitting Points")
                        # </editor-fold>

                        # ax2.legend(prop={"size": LegendFontSize},facecolor='white', loc="upper right").set_zorder(1000)
                        ax2.set_ylabel(FunctionName)
                        ax2.set_xlim(DicVariable[FunctionType][Variable]["FunctionAxisXMin"],
                                     DicVariable[FunctionType][Variable]["FunctionAxisXMax"])
                        ax2.set_ylim(DicVariable[FunctionType][Variable]["FunctionAxisYMin"],
                                     DicVariable[FunctionType][Variable]["FunctionAxisYMax"])
                        ax2.set_yscale(DicVariable[FunctionType][Variable]["FunctionAxisScale"])
                        # </editor-fold>

                    axs.plot(DicDftDfInterpolateValues[StageName][Type][Variable],
                                           DicDftDfInterpolateValues[StageName][Type][VariableEnergy],
                                           color=Colors[2], linestyle=LineStyle, linewidth=LineWidth,zorder=3)
                    DftPlot = axs.scatter(DicDftDf[StageName][Type][Variable],
                                              DicDftDf[StageName][Type][VariableEnergy],
                                              color=Colors[2], edgecolors=EdgeColor, linewidth=EdgeWidth, s=MarkerSize,
                                              label="DFT",zorder=4)

                    axs.plot(DicEamDfInterpolateValue[StageName][Type][Variable],
                                           DicEamDfInterpolateValue[StageName][Type][VariableEnergy],
                                           color=Colors[0], linestyle=LineStyle, linewidth=LineWidth,zorder=5)
                    EamAfterPlot = axs.scatter(DicEamDf[StageName][Type][Variable],
                                              DicEamDf[StageName][Type][VariableEnergy],
                                              color=Colors[0], edgecolors=EdgeColor, linewidth=EdgeWidth, s=MarkerSize,
                                              label="EAM: After",zorder=6)


                    axs.plot(DicEamDfInterpolateValue["Egt"][Type][Variable],
                                           DicEamDfInterpolateValue["Egt"][Type][VariableEnergy],
                                           color=Colors[1], linestyle=LineStyle, linewidth=LineWidth,zorder=7)
                    EamBeforePlot = axs.scatter(DicEamDf["Egt"][Type][Variable],
                                              DicEamDf["Egt"][Type][VariableEnergy],
                                              color=Colors[1], edgecolors=EdgeColor, linewidth=EdgeWidth, s=MarkerSize,
                                              label="EAM: Before",zorder=8)


                    # axs.set_title(Type, fontsize=FontSize)
                    axs.set_xlabel(Variable)
                    axs.set_ylabel("Excess Energy (Ev)")
                    axs.set_xlim(DicVariable[FunctionType][Variable]["EamAxisXMin"],
                                               DicVariable[FunctionType][Variable]["EamAxisXMax"])
                    axs.set_ylim(DicVariable[FunctionType][Variable]["EamAxisYMin"],
                                               DicVariable[FunctionType][Variable]["EamAxisYMax"])
                    # axs[0].set_xscale("log")
                    # print(DicVariable[FunctionType][Variable])
                    axs.set_yscale(DicVariable[FunctionType][Variable]["EamAxisScale"])
                    axs.spines[["right", "top"]].set_visible(True)
                    # axs[0].legend(prop={"size": 12})
                    # plt.grid()
                    h, l = np.hstack([axis.get_legend_handles_labels()
                                      for axis in axs.figure.axes
                                      if axis.bbox.bounds == axs.bbox.bounds]).tolist()
                    plt.legend(handles=h, labels=l, loc='upper right',frameon=False)
                    # axs[0][TypeCount].legend(prop={"size": LegendFontSize},facecolor='white', loc="upper left").set_zorder(1000)

                    TypeCount += 0#1
                    # axs[0][Columns - 1].legend(prop={"size": LegendFontSize},facecolor='white', loc="upper left").set_zorder(1000)
                    plt.tight_layout()
                    # plt.xlim(-0.5,5)
                    plt.savefig(Date + "-" + Title + "-Presentation.png")
                    if PlottingShow:
                        plt.show()
                    plt.close()
                # </editor-fold>
        # </editor-fold>
        Plotting = False
        PlottingShow = False

        Plotting = False
        PlottingShow = False
        # <editor-fold desc="EamDftFunction-1x3">
        if PrintSectionTitle_V: print("EamDftFunction-1x3")
        if Plotting:
            # if StageName != "Original":
            if StageName == "EgtSatZblPpmEpm-4" or StageName == "EgtSatZblPpm-4":
                # print("StageName is : " + str(StageName))
                Title = StageName_V + "-" + FunctionType + "-" + "DftRhoEnergy"
                LineWidth = 3
                FontSize = 20
                LegendFontSize = 10
                LineStyle = "-"
                ColorStyle = "red"
                EdgeColor = None#"black"
                MarkerSize = 40
                EdgeWidth = 1
                Rows = 1
                Columns = len(TypeList)
                fig, axs = plt.subplots(Rows, Columns, squeeze=False, figsize=(4 * Columns, 4))
                TypeCount = 0
                for Type in TypeList:

                    if (StageNamePure == "EgtSatZblPpm") or (StageNamePure == "EgtSatZblPpmEpm"):

                        # <editor-fold desc="Function">
                        ax2 = axs[0][TypeCount].twinx()
                        axs[0][TypeCount].set_zorder(1)  # default zorder is 0 for ax1 and ax2
                        axs[0][TypeCount].patch.set_visible(False)  # prevents ax1 from hiding ax2
                        # ax2.plot(FittingNumpyDist, FittingNumpyInterpolateValue, color=Colors[3])
                        # ax2.scatter(FittingNumpy[:, 0], FittingNumpy[:, 1], color=Colors[3], label=Variable + "-" + FunctionName + ": FittingNumpy")
                        # print(FittingNumpy)
                        FuncOriginalPlot = ax2.plot(DicPotential["Egt"][VariableDist],
                                    DicPotential["Egt"][FunctionName], color=Colors[4],
                                    label=DicVariable[FunctionType][Variable]["FunctionName"] + ": Before",zorder=1)
                        FuncAfterPlot = ax2.plot(DicPotential[StageName_V][VariableDist],
                                    DicPotential[StageName_V][FunctionName], color=Colors[6],
                                    label=DicVariable[FunctionType][Variable]["FunctionName"] + ": After",zorder=2)
                        # ax2.plot(RhoDist, RhoFOriginalInterpolateValues, color=Colors[4])

                        # <editor-fold desc="Fitting Points">
                        # for row in range(FittingNumpyXvalue_V.shape[0]):
                        #     FittingPoint = FittingNumpyXvalue_V[row]
                        #     ax2.axvline(x=FittingPoint, linestyle="dashed", color=Colors[3], linewidth=1)
                        # ax2.axvline(x=FittingPoint, linestyle="dashed", color=Colors[3], linewidth=1,
                        #             label="Fitting Points")
                        # </editor-fold>

                        # ax2.legend(prop={"size": LegendFontSize},facecolor='white', loc="upper right").set_zorder(1000)
                        ax2.set_ylabel(FunctionName)
                        ax2.set_xlim(DicVariable[FunctionType][Variable]["FunctionAxisXMin"],
                                     DicVariable[FunctionType][Variable]["FunctionAxisXMax"])
                        ax2.set_ylim(DicVariable[FunctionType][Variable]["FunctionAxisYMin"],
                                     DicVariable[FunctionType][Variable]["FunctionAxisYMax"])
                        ax2.set_yscale(DicVariable[FunctionType][Variable]["FunctionAxisScale"])
                        # </editor-fold>

                    axs[0][TypeCount].plot(DicDftDfInterpolateValues[StageName][Type][Variable],
                                           DicDftDfInterpolateValues[StageName][Type][VariableEnergy],
                                           color=Colors[2], linestyle=LineStyle, linewidth=LineWidth,zorder=3)
                    DftPlot = axs[0][TypeCount].scatter(DicDftDf[StageName][Type][Variable],
                                              DicDftDf[StageName][Type][VariableEnergy],
                                              color=Colors[2], edgecolors=EdgeColor, linewidth=EdgeWidth, s=MarkerSize,
                                              label="DFT",zorder=4)

                    axs[0][TypeCount].plot(DicEamDfInterpolateValue[StageName][Type][Variable],
                                           DicEamDfInterpolateValue[StageName][Type][VariableEnergy],
                                           color=Colors[0], linestyle=LineStyle, linewidth=LineWidth,zorder=5)
                    EamAfterPlot = axs[0][TypeCount].scatter(DicEamDf[StageName][Type][Variable],
                                              DicEamDf[StageName][Type][VariableEnergy],
                                              color=Colors[0], edgecolors=EdgeColor, linewidth=EdgeWidth, s=MarkerSize,
                                              label="EAM: After",zorder=6)


                    axs[0][TypeCount].plot(DicEamDfInterpolateValue["Egt"][Type][Variable],
                                           DicEamDfInterpolateValue["Egt"][Type][VariableEnergy],
                                           color=Colors[1], linestyle=LineStyle, linewidth=LineWidth,zorder=7)
                    EamBeforePlot = axs[0][TypeCount].scatter(DicEamDf["Egt"][Type][Variable],
                                              DicEamDf["Egt"][Type][VariableEnergy],
                                              color=Colors[1], edgecolors=EdgeColor, linewidth=EdgeWidth, s=MarkerSize,
                                              label="EAM: Before",zorder=8)


                    # axs[0][TypeCount].set_title(Type, fontsize=FontSize)
                    axs[0][TypeCount].set_xlabel(Variable)
                    axs[0][TypeCount].set_ylabel("Excess Energy (Ev)")
                    axs[0][TypeCount].set_xlim(DicVariable[FunctionType][Variable]["EamAxisXMin"],
                                               DicVariable[FunctionType][Variable]["EamAxisXMax"])
                    axs[0][TypeCount].set_ylim(DicVariable[FunctionType][Variable]["EamAxisYMin"],
                                               DicVariable[FunctionType][Variable]["EamAxisYMax"])
                    # axs[0].set_xscale("log")
                    # print(DicVariable[FunctionType][Variable])
                    axs[0][TypeCount].set_yscale(DicVariable[FunctionType][Variable]["EamAxisScale"])
                    axs[0][TypeCount].spines[["right", "top"]].set_visible(True)
                    # axs[0].legend(prop={"size": 12})
                    # plt.grid()
                    h, l = np.hstack([axis.get_legend_handles_labels()
                                      for axis in axs[0][TypeCount].figure.axes
                                      if axis.bbox.bounds == axs[0][TypeCount].bbox.bounds]).tolist()
                    plt.legend(handles=h, labels=l, loc='upper left',frameon=False)
                    # axs.legend(prop={"size": LegendFontSize},facecolor='white', loc="upper left").set_zorder(1000)

                    TypeCount += 1
                # axs[0][Columns - 1].legend(prop={"size": LegendFontSize},facecolor='white', loc="upper left").set_zorder(1000)
                plt.tight_layout()
                # plt.xlim(-0.5,5)
                plt.savefig(Date + "-" + Title + "-Presentation.png")
                if PlottingShow:
                    plt.show()
                plt.close()
                # </editor-fold>
        # </editor-fold>
        Plotting = False
        PlottingShow = False

    return DicPotential, DicEamCritical

# </editor-fold>

# <editor-fold desc="Make Video">
# OpenCV VideoWriter
def FrameToVideo(Address_V, TimesCalled, Type_V = "A" , MaskName_V =  "Video", Format_V = "png", FrameSize_V = (640, 480),fps = 4, Date_V = Date, StageName_V = StageName):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # ParentDirectoryTwo = Path(Address_V).resolve().parents[1]
    # print(ParentDirectoryTwo)
    FormatMask = "*." + Format_V
    FormatTail = "." + Format_V
    ExtensionCounter = len(glob.glob1(Address_V, FormatMask))
    NumOfFrames = ExtensionCounter
    VideoPath = Address_V + "/" + MaskName_V + ".avi"
    # print("Video Saved in :")
    # print(VideoPath)

    RefFrameAddress = Address_V + "/" + Date_V + "-CostFunc-" + StageName_V + "-" + Type_V + "-1.png"
    RefFramePath = Path(RefFrameAddress)
    if RefFramePath.is_file():
        # print(RefFrameAddress)
        RefFrameRead = cv2.imread(RefFrameAddress)
        height, width, layers = RefFrameRead.shape
        FrameSize = (width, height)
    else:
        print("No RefFrame")
        FrameSize = FrameSize_V

    video_out = cv2.VideoWriter(VideoPath, fourcc, fps, FrameSize)

    for i in range(1,TimesCalled+1):
        FileAddress = Address_V + "/" + MaskName_V + "-" + str(i) + ".png"
        FileAddress = str(pathlib.PureWindowsPath(FileAddress))
        # print(FileAddress)
        frame = cv2.imread(FileAddress)
        if frame is not None:
            video_out.write(frame)
        else:
            print("Error: Unable to read file")

    video_out.release()
    cv2.destroyAllWindows()
    # os.system("pause")
# </editor-fold>

# </editor-fold>

# </editor-fold>

# <editor-fold desc="######################################## Minimization">
print("######################################## Minimization")

# <editor-fold desc="**********  Calculation">
print("**********  Calculation")
BoxInitial = np.array([1,1.7324,1.5980])
PotList = ["MtpPd"] # ["M2","M2R","M3","M3R","BMD192","BMD192R"]
EquilibriumDf = pd.DataFrame(columns=["Potential","Num","A1X","A2Y","A3Z","TotalEnergy","NumberOfAtoms","LatticeConstant","CohesiveEnergy",
                                      "PressureX","PressureY","PressureZ",
                                      "Volume","Mass","Density"])
Active = True
if Active:
    # <editor-fold desc="Run">
    IterationDf = pd.DataFrame(columns=['Num','A1X','A2Y','A3Z','PressureX','PressureY','PressureZ','PressureSum','Potential'])

    for Pot in PotList:
        print(Pot)

        Directory = CurrentDirectory + "/" + Pot
        # print(Directory)

        # if os.path.exists(Directory):
        #     os.remove(Directory)

        if not os.path.exists(Directory):
            os.makedirs(Directory)

        Arguments = (
            Pot,
            Directory
        )

        pressure_sums = []

        # def callbackF(Input_V):
        #     global Num
        #     global IterationDf
        #     print(str(Num), str(Input_V[0]), str(Input_V[1]), str(Input_V[2]))
        #     IterationNew = pd.DataFrame({'Num': [Num], 'X': [Input_V[0]], 'Y': [Input_V[1]], 'Z': [Input_V[2]]})
        #     IterationNew["Potential"] = Pot
        #     IterationDf = pd.concat((IterationDf, IterationNew), ignore_index=True)


        Num = 0
        PressureMinimize = minimize(EquilibriumFunc, BoxInitial,args=Arguments, method="Nelder-Mead",
                                    options={"disp": True, "maxiter": MinimizeMaxiter},
                                    tol=MinimizeTol,
                                    # callback=callbackF
                                    )

        EquilibriumNewAddress = Directory + "/" + "Equilibrium.csv"
        EquilibriumNew = pd.read_csv(EquilibriumNewAddress)
        # EquilibriumNew["Potential"] = Pot
        EquilibriumDf = pd.concat((EquilibriumDf, EquilibriumNew), ignore_index=True)
    # </editor-fold>

    # <editor-fold desc="Save">
    EquilibriumDf.to_csv(Date + "-" + "Equilibrium.csv", index=False)
    IterationDf.to_csv(Date + "-" + "IterationDf.csv", index=False)
    # </editor-fold>

else:
    # <editor-fold desc="Load">
    EquilibriumDf = pd.read_csv(Date + "-" + "Equilibrium.csv")
    # print(EquilibriumDf)
    IterationDf = pd.read_csv(Date + "-" + "IterationDf.csv")
    # </editor-fold>

# </editor-fold>

# <editor-fold desc="**********  Analysis: Energy and Pressure">
print("**********  Analysis")

# <editor-fold desc="Plot - IterationDfPressureSum">
Active = False
if Active:
    Title = "IterationDfPressureSum"

    plt.figure(figsize=(8, 8))
    sns.lineplot(
        data=IterationDf,
        x='Num',
        y='PressureSum',
        hue='Potential',
        palette=Colors
    )

    plt.xlabel("Iteration")
    plt.ylabel("Total Pressure (bars)")

    plt.xlim(0, 80)
    # plt.xscale("log")
    # plt.yscale("log")
    plt.savefig(Date + "-" + Title)
    if PlottingShow: plt.show()
# </editor-fold>

# <editor-fold desc="Plot - IterationDfPressure">
Active = False
if Active:
    Title = "IterationDfPressure"
    IterationDfMelted = pd.melt(IterationDf, id_vars=['Num', 'Potential'],
                        # value_vars=['A1X', 'A2Y', 'A3Z', 'PressureX', 'PressureY', 'PressureZ', 'PressureSum'],
                        value_vars=['PressureX', 'PressureY', 'PressureZ'],
                        var_name='Direction', value_name='Pressure')
    print(IterationDfMelted)

    plt.figure(figsize=(8, 8))
    IterationDfMeltedPlot = sns.lineplot(
        data=IterationDfMelted,
        x='Num',
        y='Pressure',
        hue='Potential',
        style='Direction',
        palette=Colors
    )

    plt.xlabel("Iteration")
    plt.ylabel("Pressure (bars)")

    plt.xlim(0, 80)
    # plt.xscale("log")
    # plt.yscale("log")
    plt.legend(loc='upper right')
    plt.savefig(Date + "-" + Title)
    if PlottingShow: plt.show()
# </editor-fold>

# <editor-fold desc="Plot - CohesiveEnergy">
Active = False
if Active:
    # plt.figure(figsize=(10, 10))
    EquilibriumDf['CohesiveEnergy']=EquilibriumDf['CohesiveEnergy'].abs()
    Title = "CohesiveEnergy"
    Plot = sns.catplot(data=EquilibriumDf, x="Potential", y="CohesiveEnergy", kind="bar",
                       height = 6, aspect = 1,
                       # errorbar=None,
                       # estimator=np.mean, capsize=.2,
                       # palette=[colors[2], colors[1], colors[0]],
                       # legend=True,  # , linewidth = 1.5)
                       # col="Potential", col_wrap=1,
                       # hue="Potential"#, hue_order=["O", "C", "Bo"]
                       palette=Colors)
    # plt.title("Slope")
    Plot.set_xlabels("Potential", clear_inner=False)
    Plot.set_ylabels("Energy per Atom (eV)", clear_inner=False)
    plt.xticks(rotation=0)

    # plt.xlim(0, 50)
    # plt.xscale("log")
    # plt.yscale("log")
    plt.savefig(Date + "-" + Title)
    if PlottingShow: plt.show()
# </editor-fold>

# </editor-fold>

# <editor-fold desc="**********  Analysis: Lattice Constant">
print("**********  Analysis: Lattice Constant")

# <editor-fold desc="Dump Extraction ">
Active = True
if Active:
    DumpDfColumns = ["Potential", "FileName", "Lattice",
                     "BoxX1", "BoxX2", "BoxY1", "BoxY2", "BoxZ1", "BoxZ2",
                     "id", "type", "x", "y", "z", "vx", "vy",  "vz", "c_eng"]

    DumpDf = pd.DataFrame(columns=DumpDfColumns)
    for root, dirs, files in os.walk(CurrentDirectory, topdown=False):  # open the files
        for name in files:
            FileName = "Final.dump"
            if FileName in name:  # target the dump files
                Address = root + "/" + name
                Address = pathlib.PureWindowsPath(Address)
                Parent = root.split("\\")[-1]
                DumpNew = DumpExplorer(Address, BatchRead_V="None")
                # print(DumpNew)
                DumpNewDfColumns = ["FileName","Lattice","BoxX1","BoxX2","BoxY1","BoxY2","BoxZ1","BoxZ2",
                                    "id", "type", "x", "y", "z", "vx", "vy", "vz", "c_eng"]

                DumpNewDf = pd.DataFrame(DumpNew, columns=DumpNewDfColumns)
                DumpNewDf["FileName"] = FileName
                DumpNewDf["Potential"] = Parent
                DumpDf = pd.concat((DumpDf, DumpNewDf), ignore_index=True)

    # <editor-fold desc="Save">
    DumpDf.sort_values(by=["Potential"])
    DumpDf.to_csv(Date + "-" + "DumpDf.csv", index=False)
    # </editor-fold>

else:
    # <editor-fold desc="Load">
    DumpDf = pd.read_csv(Date + "-" + "DumpDf.csv")
    # </editor-fold>
# </editor-fold>

# <editor-fold desc="Dataframe">
print("Dataframe")
LatticeConstant = DumpDf

def SliceMiddleZ(group):
    LatticeConstantUnique = sorted(group['z'].unique())
    LatticeConstantUniqueZ = LatticeConstantUnique[len(LatticeConstantUnique) // 2]
    return group[group['z'] == LatticeConstantUniqueZ]

def SliceMiddleY(group):
    LatticeConstantUnique = sorted(group['y'].unique())
    LatticeConstantUniqueZ = LatticeConstantUnique[len(LatticeConstantUnique) // 2]
    return group[group['y'] == LatticeConstantUniqueZ]

def SliceMiddleX(group):
    LatticeConstantUnique = sorted(group['x'].unique())
    LatticeConstantUniqueZ = LatticeConstantUnique[len(LatticeConstantUnique) // 2]
    return group[group['x'] == LatticeConstantUniqueZ]

def AverageDistanceX(group):
    # Find the atoms with minimum and maximum x values
    Min = group.loc[group['x'].idxmin()]['x']
    Max = group.loc[group['x'].idxmax()]['x']
    # print(min)
    # print(max)
    distance = np.abs(Max - Min)
    # print(len(group))
    result = distance / (len(group)-1)
    # print(result)
    group['LatticeDistX'] = result
    return group

def AverageDistanceZ(group):
    # Find the atoms with minimum and maximum x values
    Min = group.loc[group['z'].idxmin()]['z']
    Max = group.loc[group['z'].idxmax()]['z']
    # print(min)
    # print(max)
    distance = np.abs(Max - Min)
    result = distance / (len(group)-1)
    # print(result)
    group['LatticeDistZ'] = result
    return group

LatticeConstantX = LatticeConstant.groupby(['Potential'], group_keys=False).apply(SliceMiddleZ)
LatticeConstantX = LatticeConstantX.groupby(['Potential'], group_keys=False).apply(SliceMiddleY)

LatticeConstantZ = LatticeConstant.groupby(['Potential'], group_keys=False).apply(SliceMiddleY)
LatticeConstantZ = LatticeConstantZ.groupby(['Potential'], group_keys=False).apply(SliceMiddleX)

LatticeConstantX = LatticeConstantX.groupby(['Potential'], group_keys=False).apply(AverageDistanceX)
LatticeConstantX = LatticeConstantX.drop(columns=['id', 'type', 'x', 'y', 'z', 'c_eng'])
LatticeConstantX = LatticeConstantX.drop_duplicates(subset=['Potential'])

LatticeConstantZ = LatticeConstantZ.groupby(['Potential'], group_keys=False).apply(AverageDistanceZ)
LatticeConstantZ = LatticeConstantZ.drop(columns=['id', 'type', 'x', 'y', 'z', 'c_eng'])
LatticeConstantZ = LatticeConstantZ.drop_duplicates(subset=['Potential'])

LatticeConstant = pd.merge(LatticeConstantX, LatticeConstantZ, how='inner',
                           on=["Potential", "FileName", "Lattice", "BoxX1", "BoxX2", "BoxY1", "BoxY2", "BoxZ1", "BoxZ2"])

LatticeConstantMelt = pd.melt(LatticeConstant, id_vars=['Potential'],
                    value_vars=['LatticeDistX', 'LatticeDistZ'],
                    var_name='LatticeConstantType',
                    value_name='LatticeConstant')
LatticeConstantMelt = LatticeConstantMelt.rename(columns={"LatticeConstantType": "Direction"})
LatticeConstantMelt['Direction'] = LatticeConstantMelt['Direction'].map({'LatticeDistZ': "c", 'LatticeDistX': "a"})
# print(LatticeConstantMelt)
# </editor-fold>

# <editor-fold desc="Plot">
print("Plot")
Active = True
if Active:
    Title = "LatticeConstant"

    potentials = ["", " "]
    types = ["a", "c"]

    SpaceDf = pd.DataFrame({
    "Potential": [potential for potential in potentials for _ in types],
    "Type": types * len(potentials)
    })
    SpaceDf["FormationEnergy"] = 0

    LatticeConstantMelt = pd.concat((LatticeConstantMelt, SpaceDf), ignore_index=True)
    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax = sns.barplot(
        LatticeConstantMelt,# kind="bar",
        x="Direction", y="LatticeConstant",
        hue="Potential", hue_order=["M2", "M2R", "", "M3", "M3R", "  ", "BMD192", "BMD192R"],
        width=0.7,
        # height=6, aspect=1,
        palette=Colors,
        zorder = 3,
        # legend = False
    )

    ax.set(xlabel=None)#'Equilibrium Lattice Constant')#, fontsize=10)  # not set_label
    ax.set_ylabel('Lattice Constant (Å)', fontsize=25)

    # Update this part to rename the 'Direction' x-axis labels
    direction_labels = {'a': r'$a_0$', 'c': r'$c_0$'}  # Replace 'a' and 'c' with actual Direction categories

    # print(fig.axes)
    ax.set_xticklabels([direction_labels.get(label.get_text(), label.get_text()) for label in ax.get_xticklabels()],
                       fontsize=25)


    # plt.xlabel("SIA Type")
    # plt.ylabel("Lattice Constant")
    plt.ylim(0,5.5)

    Color = "Black"
    Alpha = 0.25
    x_labels = ax.get_xticklabels()
    # print(x_labels)
    for i, label in enumerate(x_labels):
        print(label)
        if label.get_text() == "$a_0$":
            x_c = i
            break
    DftMin = 3.223
    DftMax = 3.239
    DftRange = DftMax-DftMin
    ax.fill_between([x_c - 0.4, x_c + 0.4], DftMin, DftMax, color=Color, alpha=Alpha)
    rect = patches.Rectangle((x_c - 0.4, DftMin),  0.8, DftRange, linewidth=1.5, edgecolor='black', facecolor='none')
    ax.add_patch(rect)


    Color = "Black"
    Alpha = 0.25
    x_labels = ax.get_xticklabels()
    # print(x_labels)
    for i, label in enumerate(x_labels):
        print(label)
        if label.get_text() == "$c_0$":
            x_c = i
            break
    DftMin = 5.152
    DftMax = 5.175
    DftRange = DftMax-DftMin
    ax.fill_between([x_c - 0.4, x_c + 0.4], DftMin, DftMax, color=Color, alpha=Alpha)
    rect = patches.Rectangle((x_c - 0.4, DftMin),  0.8, DftRange, linewidth=1.5, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    legend_elements = [
        mpatches.Patch(color="b", label="M2"),
        mpatches.Patch(color="cyan", label="M2R"),
        mpatches.Patch(color=Color, label="DFT"),
        mpatches.Patch(color="darkred", label="M3"),
        mpatches.Patch(color="red", label="M3R"),
        mpatches.Patch(color="green", label="BMD192"),
        mpatches.Patch(color="lime", label="BMD192R"),
    ]

    # Add the custom legend on top
    plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, 1.00), loc='center', ncol=3, frameon=False)

    plt.tight_layout()
    plt.savefig(Date + "-" + Title)
    if PlottingShow: plt.show()
# </editor-fold>

# <editor-fold desc="Shifting">
# DumpDf["BoxX"] = DumpDf["BoxX2"] - DumpDf["BoxX1"]
# DumpDf["BoxY"] = DumpDf["BoxY2"] - DumpDf["BoxY1"]
# DumpDf["BoxZ"] = DumpDf["BoxZ2"] - DumpDf["BoxZ1"]
# DumpDf["BoxX1"] = DumpDf["BoxX1"] - DumpDf["BoxX1"]
# DumpDf["BoxY1"] = DumpDf["BoxY1"] - DumpDf["BoxY1"]
# DumpDf["BoxZ1"] = DumpDf["BoxZ1"] - DumpDf["BoxZ1"]
# DumpDf["BoxX2"] = DumpDf["BoxX2"] - DumpDf["BoxX1"]
# DumpDf["BoxY2"] = DumpDf["BoxY2"] - DumpDf["BoxY1"]
# DumpDf["BoxZ2"] = DumpDf["BoxZ2"] - DumpDf["BoxZ1"]
# DumpDf["x"] = DumpDf["x"] - DumpDf["BoxX1"]
# DumpDf["y"] = DumpDf["y"] - DumpDf["BoxY1"]
# DumpDf["z"] = DumpDf["z"] - DumpDf["BoxZ1"]

# DumpDf['DistToCorner'] = np.sqrt(DumpDf['x']**2 + DumpDf['y']**2 + DumpDf['z']**2)
#
# DumpDfGrouped = DumpDf.groupby('Potential')
#
#
# def shift_group(group):
#     # Find the atom closest to the origin
#     closest_atom = group.loc[group['DistToCorner'].idxmin()]
#
#     # Calculate the shift
#     shift_x = closest_atom['x']
#     shift_y = closest_atom['y']
#     shift_z = closest_atom['z']
#
#     # Apply the shift
#     group['x'] -= shift_x
#     group['y'] -= shift_y
#     group['z'] -= shift_z
#
#     return group
#
# DumpDf = DumpDfGrouped.apply(shift_group)
#
# # Drop the 'distance' column
# DumpDf = DumpDf.drop(columns=['DistToCorner'])
# </editor-fold>

# # <editor-fold desc="Labeling">
# def label_atoms_periodic(group):
#     # Unwrap coordinates using periodic boundary conditions
#     for axis in ['x', 'y', 'z']:
#         box_min = group[f'Box{axis.upper()}1'].iloc[0]
#         box_max = group[f'Box{axis.upper()}2'].iloc[0]
#         box_length = box_max - box_min
#         group[axis] = np.mod(group[axis] - box_min, box_length) + box_min
#
#     # Sort and label
#     group_sorted = group.sort_values(by=['x', 'y', 'z'])
#     group_sorted['Label'] = ['A', 'B', 'C', 'D'][:len(group_sorted)]
#     return group_sorted
#
# def label_atoms(group):
#     # Sort atoms based on x, y, z positions
#     group_sorted = group.sort_values(by=['x', 'y', 'z'])
#     # Assign labels A, B, C, D
#     group_sorted['Label'] = ['A', 'B', 'C', 'D']
#     return group_sorted
#
# # Apply the labeling function to each group
# DumpDf = DumpDf.groupby('Potential').apply(label_atoms).reset_index(drop=True)
# # </editor-fold>

# # <editor-fold desc="Movement measurement">
# DumpDf['Reparameterization'] = DumpDf['Potential'].str.endswith('R')
#
# DumpDf['Original'] = DumpDf.apply(
#     lambda row: row['Potential'][:-1] if row['Reparameterization'] else np.nan,
#     axis=1
# )
#
# DumpDfOriginal = DumpDf[~DumpDf['Reparameterization']]
# DumpDfOriginal = DumpDfOriginal.drop(columns=['Original'])
# DumpDfReparameterized = DumpDf[DumpDf['Reparameterization']]
# # DumpDfOriginal.to_csv(Date + "-" + "DumpDfOriginal.csv", index=False)
# DumpDfReparameterized = DumpDfReparameterized[['Original', 'id','x', 'y', 'z']]
# DumpDfReparameterized = DumpDfReparameterized.rename(columns={'Original': 'Potential'})
# # DumpDfReparameterized.to_csv(Date + "-" + "DumpDfReparameterized_filtered.csv", index=False)
#
# DumpDfMelt = pd.merge(DumpDfOriginal, DumpDfReparameterized, on=['Potential','id'], how='inner',suffixes=('', '_after'))
#
#
# def calculate_pbc_distance(group):
#     for axis in ['x', 'y', 'z']:
#         delta_col = f'delta_{axis}'
#         pbc_col = f'pbc_distance_{axis}'
#         box_col = f'Box{axis.upper()}'
#
#         # Calculate the delta and the initial PBC distance
#         group[delta_col] = group[f'{axis}_after'] - group[axis]
#         pbc_distance = (group[delta_col] + 0.5 * group[box_col]) % group[box_col] - 0.5 * group[box_col]
#
#         # Calculate the minimum distance considering periodic boundaries
#         group[pbc_col] = np.abs(
#             np.where(np.abs(pbc_distance) > 0.5 * np.abs(group[box_col]), np.abs(group[box_col]) - np.abs(pbc_distance),
#                      np.abs(pbc_distance)))
#
#     return group
#
#
# # Group by 'Potential' and 'id', and apply the PBC distance calculation
# DumpDfMelt = DumpDfMelt.groupby(['Potential', 'id']).apply(calculate_pbc_distance).reset_index(drop=True)
#
#
# # </editor-fold>

# # <editor-fold desc="Save">
# DumpDf.to_csv(Date + "-" + "DumpDf.csv", index=False)
# DumpDfMelt.to_csv(Date + "-" + "DumpDfMelt.csv", index=False)
# # </editor-fold>

# </editor-fold>

# <editor-fold desc="**********  Analysis: Density">
print("**********  Analysis: Density")

# <editor-fold desc="Plot">
print("Plot")
Active = False
if Active:
    Title = "Density"

    # fig = plt.figure(figsize=(10, 10))

    Plot = sns.catplot(
        EquilibriumDf, kind="bar",
        x="Potential", y="Density",# hue="Potential"
        height=6, aspect=1,
        palette=Colors
    )

    Plot.set_xlabels('Potential', fontsize=10)  # not set_label
    Plot.set_ylabels('Density (gram/cm3)', fontsize=10)

    plt.savefig(Date + "-" + Title)
    if PlottingShow: plt.show()
# </editor-fold>


# </editor-fold>
# </editor-fold>