#!/usr/bin/env python3

from pybest import context
from pybest.ee_eom import REOMpCCDS
from pybest.geminals import ROOpCCD, RpCCD
from pybest.linalg import DenseLinalgFactory
from pybest.modelhamiltonians.ppp_model import PPP
from pybest.occ_model import AufbauOccModel
from pybest.units import electronvolt
from pybest.ci import RCID, RCIS, RCISD
from pybest.cc import RfpCCD, RfpCCSD, RCCSD
from pybest.ci import RCIS, RCISD, RCID, RpCCDCID, RpCCDCISD, RpCCDCIS
from pybest.wrappers import RHF
from pybest.geminals import RpCCD, ROOpCCD
from pybest.cc import RpCCDLCCD, RpCCDLCCSD
from pybest.ee_eom import REOMpCCDLCCD, REOMpCCDLCCSD
from pybest.ee_eom import REOMptCCSD, REOMfpCCSD, REOMCCSD
from pybest.ee_eom import REOMpCCDS
from pybest.cc import RfpCCD, RfpCCSD
from pybest.ip_eom import RIPpCCD

# get the xyz file from pybest/src/pybest/data/test
#coord = context.get_fn("test/c28-bp86.xyz")
coord = "c60-bp86.xyz"
coords = None

# Number of sites represented as a `LinalgFactory` object (indicating the number of supported atoms).
lf = DenseLinalgFactory(60)

# Define the occupation model where `nel` is the number of C-H bonding and lone-pair electrons.
occ_model = AufbauOccModel(lf, nel=60)
orb_a = lf.create_orbital()

# t: hopping, u: e-e repulsion, k: dielectric constant, u_p=u/k, hubbard: hubbard term in ppp.
modelham = PPP(lf, occ_model, xyz_file=coord)

ppp_hamiltonian = modelham(
    parameters={
        "on_site": 0.0,
        "hopping": -2.7 * electronvolt,
        "u": 8.0 * electronvolt,
        "k": "topology",
        "hubbard": True,
    },
)

olp = modelham.compute_overlap()

# Do RHF calculation
hf = RHF(lf, occ_model)
hf_output = hf(ppp_hamiltonian, 0.0, olp, orb_a)

ccsd = RCCSD(lf, occ_model)
ccsd_output = ccsd(hf_output, ppp_hamiltonian, solver="krylov")
#
# Do REOM-CCSD(pCCD)
#
eom = REOMCCSD(lf, occ_model)
_ = eom(ppp_hamiltonian, ccsd_output, nroot=50)
