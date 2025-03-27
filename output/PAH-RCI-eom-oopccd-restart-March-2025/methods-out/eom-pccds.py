#!/usr/bin/env python3
from pybest import context
from pybest.linalg import DenseLinalgFactory
from pybest.modelhamiltonians.ppp_model import PPP
from pybest.occ_model import AufbauOccModel
from pybest.units import electronvolt
from pybest.ci import RCIS, RCISD
from pybest.wrappers import RHF
from pybest.geminals import RpCCD, ROOpCCD
from pybest.cc import RpCCDLCCD, RpCCDLCCSD
from pybest.ee_eom import REOMpCCDLCCD, REOMpCCDLCCSD
from pybest.ee_eom import REOMpCCDS
from pybest.cc import RfpCCD, RfpCCSD
from termcolor import colored
import numpy as np

# get the xyz file from pybest/src/pybest/data/test
coord = context.get_fn("test/c28.xyz")

# Number of sites represented as a `LinalgFactory` object (indicating the number of supported atoms).
lf = DenseLinalgFactory(28)

# Define the occupation model where `nel` is the number of C-H bonding and lone-pair electrons.
occ_model = AufbauOccModel(lf, nel=28)
orb_a = lf.create_orbital()

# t: hopping, u: e-e repulsion, k: dielectric constant, hubbard: hubbard term in ppp.
modelham = PPP(lf, occ_model, xyz_file=coord)

huckel_output = modelham(
    parameters={
        "on_site": 0.0,
        "hopping": -4.1 * electronvolt,
        "u": 6.0* electronvolt,
        "k": 2.0,
        "hubbard": False,
    }
)

one_body = modelham.compute_one_body()
two_body = modelham.compute_two_body()
olp = modelham.compute_overlap()

# Do a Hartree-Fock calculation
hf = RHF(lf, occ_model)
hf_output = hf(one_body, two_body, 0.0, olp, orb_a)

oopccd = ROOpCCD(lf, occ_model) 
oopccd_output = oopccd(
    one_body,
    two_body, hf_output, maxiter={"orbiter":0})

eom = REOMpCCDS(lf, occ_model)
eom_output = eom(one_body,two_body, oopccd_output, nroot=4)