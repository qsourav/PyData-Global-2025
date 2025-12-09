import os
import pandas as pd
from datetime import timedelta
from linetimer import CodeTimer, linetimer
from pycountry_convert import (
    country_name_to_country_alpha2,
    country_alpha2_to_continent_code,
)

def evaluate(df):
    if hasattr(df, "_evaluate"):
        df._evaluate()


def load_data():
    return pd.read_parquet("Online Retail.parquet")


def prep_data():
    df = load_data()
    # simply inflated the loaded data since it only contains 0.5M of samples 
    df = pd.concat([df] * int(os.environ.get("SF", 5)))
    return df


def to_region(country):
    try:
        alpha2 = country_name_to_country_alpha2(country)
        cont = country_alpha2_to_continent_code(alpha2)
        return cont
    except:
        return None

mapper = {
    "United Kingdom": "EU",
    "France": "EU",
    "Australia": "OC",
    "Netherlands": "EU",
    "Germany": "EU",
    "Norway": "EU",
    "EIRE": None,
    "Switzerland": "EU",
    "Spain": "EU",
    "Poland": "EU",
    "Portugal": "EU",
    "Italy": "EU",
    "Belgium": "EU",
    "Lithuania": "EU",
    "Japan": "AS",
    "Iceland": "EU",
    "Channel Islands": None,
    "Denmark": "EU",
    "Cyprus": "AS",
    "Sweden": "EU",
    "Austria": "EU",
    "Israel": "AS",
    "Finland": "EU",
    "Bahrain": "AS",
    "Greece": "EU",
    "Hong Kong": "AS",
    "Singapore": "AS",
    "Lebanon": "AS",
    "United Arab Emirates": "AS",
    "Saudi Arabia": "AS",
    "Czech Republic": "EU",
    "Canada": "NA",
    "Unspecified": None,
    "Brazil": "SA",
    "USA": "NA",
    "European Community": None,
    "Malta": "EU",
    "RSA": None,
}


def func_1(df):
    # StockCode wise mode of Description
    most_freq = (
        df[["StockCode", "Description"]]
        .value_counts()
        .reset_index()
        .groupby("StockCode")
        .head(1)
    )
    most_freq.columns = ["StockCode", "freq_Description", "frequency"]

    # transform: fillna(mode)
    tmp = df.merge(most_freq, on="StockCode", how="left")
    #df["Description"] = df["Description"].fillna(tmp["freq_Description"])
    df["Description"] = df["Description"].mask(
        df["Description"].isnull(), tmp["freq_Description"]
    )

    evaluate(df)
    return df


def func_2(df, days_offset=5):
    delta = timedelta(days=days_offset)
    # df = df.groupby(["CustomerID", "InvoiceNo"], as_index=False)["InvoiceDate"].first()
    df = load_data().groupby(["CustomerID", "InvoiceNo"], as_index=False)["InvoiceDate"].first()
    df = pd.concat([df] * int(os.environ.get("SF", 5))) 
    ret = (
        df.merge(df, on="CustomerID")
        .pipe(lambda m: m[abs(m["InvoiceDate_x"] - m["InvoiceDate_y"]) <= delta])
        .groupby(["CustomerID", "InvoiceNo_x"], as_index=False, sort=False)[
            "InvoiceNo_y"
        ]
        .size()
    )
    ret.columns = ["CustomerID", "InvoiceNo", "count"]
    ret = ret.sort_values("count")
    evaluate(ret)
    return ret


def func_3(df):
    ret = (
        df.assign(revenue=lambda x: x["Quantity"] * x["UnitPrice"])
        .groupby(["CustomerID", "InvoiceNo"], as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
    )
    ret.columns = ["CustomerID", "InvoiceNo", "revenue"]
    evaluate(ret)
    return ret


def func_4(df):
    # mapper = {m: to_region(m) for m in df["Country"].unique()}
    left = df.assign(region=lambda x: x["Country"].map(mapper))
    right = df.rename(
        columns=lambda c: c + "_y" if c not in ["CustomerID", "StockCode"] else c
    )
    evaluate(left)
    evaluate(right)

    ret = (
        left.merge(right, on=["CustomerID", "StockCode"])
        .pipe(lambda m: m[m["Quantity"] > 0])
        .pipe(lambda m: m[m["Quantity_y"] < 0])
        .pipe(lambda m: m[m["region"] == "NA"])
        .groupby("CustomerID")["InvoiceNo"]
        .nunique()
    )
    evaluate(ret)
    return ret


def func_5(df):
    ret = (
        df.pipe(lambda x: x[x["Quantity"] > 0])
        .pipe(lambda x: x[x["UnitPrice"] > 0])
        .assign(revenue=lambda x: x["Quantity"] * x["UnitPrice"])
        .assign(month=lambda x: x["InvoiceDate"].dt.month)
        .groupby(["Country", "month"], as_index=False)["revenue"]
        .sum()
    )
    evaluate(ret)
    return ret


df = prep_data()
evaluate(df)

m_name = getattr(pd.__spec__.loader, "fast_lib", pd.__name__).split(".")[0]
for fn in [func_1, func_2, func_3, func_4, func_5]:
    with CodeTimer(
        name=f"[{m_name}] Execution time of {fn.__name__}: ",
        unit="s",
    ) as timer:
        try:
            fn(df)
        except Exception as e:
            print(f"Failed with {e}")

