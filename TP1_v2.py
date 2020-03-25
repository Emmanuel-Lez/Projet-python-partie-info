# %% Import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt 
# Import documents excel
# %% Point 1: Charger données  climatiques
f1=pd.read_excel("data.xlsx", sheet_name="climat_horaire_99")
f2=pd.read_excel("data.xlsx", sheet_name="climat_horaire_00")
f3=pd.read_excel("data.xlsx", sheet_name="climat_horaire_01")

# Ajout de la variable elapsed
f1["elapsed"] = f1["JJ"] - 1
f2["elapsed"] = f2["JJ"] - 1 + 365
f3["elapsed"] = f3["JJ"] - 1 + 731

# Ajout de la variable année
f1["année"]=1999
f2["année"]=2000
f3["année"]=2001

# Grouper les 3 dataframes en 1: concat df (cdf)
cdf=pd.concat([f1,f2,f3],sort=True)

# Ajout de la variable date
date_ref=dt.datetime(1999,1,1,0,0,0)
cdf["date"]=date_ref+pd.TimedeltaIndex(cdf["elapsed"],unit="D")

# Grouper concat_data par date 
cdf.groupby(["date"])

# Moyenne journalière de la Température
t_mean=cdf[["temperature"]].groupby(cdf.date.dt.date).mean()
# Moyenne journalière de la Vitesse du vent
v_mean=cdf[["vitesse_vent"]].groupby(cdf.date.dt.date).mean()
# Moyenne journalière de l'Humidité relative
h_mean=cdf[["hum_rel"]].groupby(cdf.date.dt.date).mean()
# Moyenne journalière du Déficit de saturation
ds_mean=cdf[["DS"]].groupby(cdf.date.dt.date).mean()

# Cumul journalier de l'irradiance
i_sum=cdf[["rayt_W"]].groupby(cdf.date.dt.date).sum()
# Cumul journalier des précipitation
p_sum=cdf[["precipitation"]].groupby(cdf.date.dt.date).sum()

# Ajout de la variable rayt_J
cdf["rayt_J"]=cdf["rayt_W"]*0.0036

# Renommer df final par climate_by_day
climate_by_day=cdf
# Enregistrer sous un dossier csv appelé climate_by_day.csv
climate_by_day.to_csv('climate_by_day.csv')

# %% Point 2: Charger données de flux de sève pour chaque année

sfd_f1 = pd.read_excel("data.xlsx", sheet_name="SFD_1999")
sfd_f2 = pd.read_excel("data.xlsx", sheet_name="SFD_2000")
sfd_f3 = pd.read_excel("data.xlsx", sheet_name="SFD_2001")

# Faire la moyenne de toutes ces variables pour chaque année

sfd_by_day_mean1 = (sfd_f1[["jour","E153", "E159", "E161","T13","T21","T22"]].groupby(["jour"]).mean())
sfd_by_day_mean2 = (sfd_f2[["jour","E153", "E159", "E161","T13","T21","T22"]].groupby(["jour"]).mean())
sfd_by_day_mean3 = (sfd_f3[["jour","E153", "E159", "E161","T13","T21","T22"]].groupby(["jour"]).mean())
sfd_by_day=pd.concat([sfd_by_day_mean1, sfd_by_day_mean2,sfd_by_day_mean3])
sfd_by_day.to_csv ('sfd_by_day.csv')

# %% Point 3: Charger Données d'humidité 

humid= pd.read_excel("data.xlsx",sheet_name="Hum_Vol_Sol")
moisture_sel = humid[["T_45cm", "E_45cm"]]

# %% Point 4: Charger Pluie_Au_Sol

rain_sel=pd.read_excel("data.xlsx", sheet_name="Pluie_Au_Sol")

# %% Point 5: Assembler les 4 dataframes
df=pd.concat([climate_by_day,sfd_by_day,moisture_sel,rain_sel],sort=True)

# %% Point 6: Ajout de la variable et_0

def get_evapotranspiration(df):
    """
    This function gets a dataframe as input and returns the same dataframe with a "et0" added variable, which represents the evapotranspiration [mm/day]. This function will only work if the following variables are present in the input dataframe:
        * "DS": Vapour pressure deficit [hPA]
        * "rayt_J": Incoming solar radiation [MJ/m^2]
        * "temperature": Temperature [°C]
        * "jour": Number of the day in the year
        * "vitesse_vent": Wind speed [m/s]
    All those variables must be given as daily values.
    """

    # --- Check variables presence ---------------------------------------------
    var_in = ["DS", "rayt_J", "temperature", "jour", "vitesse_vent"]
    for var in var_in:
        if var not in df.columns:
            raise ValueError(f"'{var}' variable was not found in the input dataframe")

    # --- Parameters -----------------------------------------------------------
    ele = 150  # elevation above sea level
    degree_north = 50  # Belgium longitude (degrees)
    minute_north = 50  # Belgium longitude (minutes)

    # --- Change units ---------------------------------------------------------
    # Vapour pressure deficit [hPA] -> [kPA]
    df["DS"] = df["DS"] / 10

    # --- Pressure -------------------------------------------------------------
    # Saturated vapour pressure, kPa
    esat = 0.6108 * np.exp((17.27 * df["temperature"]) / (df["temperature"] + 237.5))
    # Actual vapour pressure, kPa
    eact = esat - df["DS"]

    # --- Extraterestrial radiation (ra) [MJ/m2/day] ---------------------------
    gsc = 0.082  # Solar constant
    dec_deg = degree_north + minute_north / 60  # Coordinates Brussels North (50° 50' N)
    phi = dec_deg / 180 * np.pi  # Coordinates in radians
    dr = 1 + 0.033 * np.cos(
        2 * np.pi * df["jour"] / 365
    )  # Distance earth sun, Eqn 23 of Allen et al. (1998)
    delta = 0.409 * np.sin(
        2 * np.pi * df["jour"] / 365 - 1.39
    )  # Solar declination, Eqn 24 of Allen et al. (1998)
    omega = np.arccos(-np.tan(phi) * np.tan(delta))  # Eqn 25 of Allen et al. (1998)
    ra = (
        24
        * 60
        * gsc
        * dr
        / np.pi
        * (
            omega * np.sin(phi) * np.sin(delta)
            + np.cos(phi) * np.cos(delta) * np.sin(omega)
        )
    )  # Eqn 21 of Allen et al. (1998)
    ra[ra < 0] = 0

    # --- Clear sky solar radiation [MJ/m2/day] --------------------------------
    rso = (0.75 + 2e-5 * ele) * ra  # Eqn 37 of Allen et al. (1998)

    # --- Net short wave radiation (albedo = 0.23) [MJ/m2/day] -----------------
    rns = 0.77 * df["rayt_J"]  # Eqn 38 of Allen et al. (1998)

    # --- Long wave radiation [MJ/m2/day] --------------------------------------
    stef_bol = 4.903e-9  # Stefa-Boltzman constant [MJ/m2/day]
    rnl = (
        stef_bol
        * ((df["temperature"] + 273.16) ** 4 / 2)
        * (0.34 - 0.14 * np.sqrt(eact))
        * (1.35 * (df["rayt_J"] / rso) - 0.35)
    )  # Eqn 39 of Allen et al. (1998)
    rnl[rnl < 0] = 0

    # --- Net radiation [MJ/m2/day] --------------------------------------------
    rn = rns - rnl  # Eqn 40 of Allen et al. (1998)

    # --- Slope of saturation deficit curve ------------------------------------
    Delta = (
        4098
        * (0.6108 * np.exp((17.27 * df["temperature"]) / (df["temperature"] + 237.3)))
        / ((df["temperature"] + 237.3) ** 2)
    )  # Eqn 13 of Allen et al. (1998)

    # --- Psychrometric constant (gamma) ---------------------------------------
    air_pres = 101.3 * (
        (293 - 0.0065 * ele / 293) ** 5.26
    )  # air pressure (kPa), Eqn 7 of Allen et al. (1998)
    cp = 1.013e-3  # specific heat at constant pressure [MJ/kg/°C]
    eps = 0.622  # ratio molecular weight of water vapour/dry air
    lam = 2.45  # latent heat of vaporization [MJ/kg]
    gamma = (cp * air_pres) / (eps * lam)  # Eqn 8 of Allen et al. (1998)

    # --- Evapotranspiration [mm/day] ------------------------------------------
    et0 = (
        (0.408 * Delta * rn)
        + (gamma * 900 / (df["temperature"] + 273) * df["vitesse_vent"] * df["DS"])
    ) / (
        Delta + gamma * (1 + 0.34 * df["vitesse_vent"])
    )  # Eqn 6 of Allen et al. (1998)

    # Add computed evapotranspiration to the dataframe
    df["et0"] = et0

    return df


# %% Point 7: Sauvegarder df sous data_full.csv
    
df.to_csv('data_full.csv')
print(df)

# %% Point 8: 



































# %% Variable évapotranspiration
import numpy as np
import pandas as pd


