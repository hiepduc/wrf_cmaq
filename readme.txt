To calculate PM2.5 from QRF-CMAQ output files, look for the formula in the species definition file in CMAQ

For example, for cb6r3_ae6_aq Chemical Mechanism the formula is:

(ASO4I[1]+ANO3I[1]+ANH4I[1]+ANAI[1]+ACLI[1]+AECI[1]+ALVOO1I[1] + ALVOO2I[1] + ASVOO1I[1] + ASVOO2I[1] +

ALVPO1I[1] + ASVPO1I[1] + ASVPO2I[1]+ AOTHRI[1])*PM25AT[2]+ (ASO4J[1]+ANO3J[1]+ANH4J[1]+ANAJ[1]+ACLJ[1]+

AECJ[1]+ AXYL1J[1] + AXYL2J[1] + AXYL3J[1] + ATOL1J[1] +ATOL2J[1] + ATOL3J[1] + ABNZ1J[1] + ABNZ2J[1]

+ABNZ3J[1] + AISO1J[1] + AISO2J[1] + AISO3J[1] +ATRP1J[1] + ATRP2J[1] + ASQTJ[1] + AALK1J[1] +

AALK2J[1] + APAH1J[1] + APAH2J[1] + APAH3J[1] +AORGCJ[1] + AOLGBJ[1] + AOLGAJ[1]+ALVOO1J[1] + ALVOO2J[1]

ASVOO1J[1] + ASVOO2J[1]+ASVOO3J[1] + APCSOJ[1] + ALVPO1J[1] + ASVPO1J[1] + ASVPO2J[1] +ASVPO3J[1] +
AIVPO1J[1]+ AOTHRJ[1]+AFEJ[1]+ASIJ[1]+ ATIJ[1]+ACAJ[1]+AMGJ[1]+AMNJ[1]+AALJ[1]+AKJ[1])* PM25AC[2]+(ASOIL[1]+

ACORS[1]+ASEACAT[1]+ACLK[1]+ASO4K[1] +ANO3K[1]+ANH4K[1])* PM25CO[2]

The term “[1]” means from CCTM_ACONC* file ant “[2]” is from CCT_APMDIAG* file.

Therefore you need to open both the ACONC file and the APMDIAG (diagnostic file)

