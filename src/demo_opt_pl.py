import os
import polars as pl
from datetime import timedelta
from linetimer import CodeTimer, linetimer
from pycountry_convert import (
    country_name_to_country_alpha2,
    country_alpha2_to_continent_code,
)


def load_data():
    return pl.scan_parquet("Online Retail.parquet")


def prep_data():
    df = load_data()
    # simply inflated the loaded data since it only contains 0.5M of samples
    df = pl.concat([df] * int(os.environ.get("SF", 5)))
    return df.collect().lazy()


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
    modes = (
        df.filter(pl.col("Description").is_not_null())
        .group_by("StockCode")
        .agg(pl.col("Description").mode().first().alias("mode_desc"))
    )

    ret = (
        df.join(modes, on="StockCode", how="left")
        .with_columns(pl.col("Description").fill_null(pl.col("mode_desc")))
        .drop("mode_desc")
    )

    return ret.collect()


def func_2(df, days_offset=5):
    # df = df.group_by(["CustomerID", "InvoiceNo"]).agg(pl.col("InvoiceDate").first())
    df = (
        load_data()
        .group_by(["CustomerID", "InvoiceNo"])
        .agg(pl.col("InvoiceDate").first())
    )
    df = pl.concat([df] * int(os.environ.get("SF", 5)))

    delta = timedelta(days=days_offset)
    ret = (
        df.join(df, on="CustomerID", suffix="_y")
        .filter((pl.col("InvoiceDate") - pl.col("InvoiceDate_y")).abs() <= delta)
        .group_by(["CustomerID", "InvoiceNo"])
        .agg(pl.col("InvoiceNo_y").count().alias("count"))
        .sort("count")
    )
    return ret.collect()


def func_3(df):
    ret = (
        df.group_by(["CustomerID", "InvoiceNo"])
        .agg((pl.col("Quantity") * pl.col("UnitPrice")).sum().alias("revenue"))
        .sort("revenue", descending=True)
    )
    return ret.collect()


def func_4(df):
    # left = df.with_columns(
    #    pl.col("Country")
    #    .map_elements(to_region, return_dtype=pl.String)
    #    .alias("region")
    # )
    # right = df.rename(
    #    {c: f"{c}_y" if c not in ["CustomerID", "StockCode"] else c for c in df.columns}
    # )
    left = df.with_columns(pl.col("Country").replace(mapper).alias("region"))
    right = df.rename(
        {
            c: f"{c}_y" if c not in ["CustomerID", "StockCode"] else c
            for c in df.collect_schema().names()
        }
    )

    ret = (
        left.join(right, on=["CustomerID", "StockCode"])
        .filter(pl.col("Quantity") > 0)
        .filter(pl.col("Quantity_y") < 0)
        .filter(pl.col("region") == "NA")
        .group_by("CustomerID")
        .agg(pl.col("InvoiceNo").n_unique())
    )
    return ret.collect()


def func_5(df):
    ret = (
        df.filter(pl.col("Quantity") > 0)
        .filter(pl.col("UnitPrice") > 0)
        .group_by(["Country", pl.col("InvoiceDate").dt.month().alias("month")])
        .agg((pl.col("Quantity") * pl.col("UnitPrice")).sum())
    )
    return ret.collect()


df = prep_data()

for fn in [func_1, func_2, func_3, func_4, func_5]:
    with CodeTimer(
        name=f"[{pl.__name__}] Execution time of {fn.__name__}: ",
        unit="s",
    ) as timer:
        try:
            fn(df)
        except Exception as e:
            print(f"Failed with {e}")
