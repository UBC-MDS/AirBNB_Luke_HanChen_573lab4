# authors: Luke Yang, HanChen Wang
# date: 2022-12-10

"""This script read the preprocessed data files and performs exploratory data analysis. Saves the resulting figures into .png files and tables as .csv files in the output folder.

Usage: data_eda.py --traindata=<traindata> --output=<output> ...

Arguments:
  --traindata=<traindata>        path of the raw training data
  --output=<output>              folder that stores the generated plots

Make sure you call this script in the repo's root path.
Example:

python src/data_eda.py --traindata=data/processed/train_cleaned.csv --output=results/eda_results/

"""

from docopt import docopt
import numpy as np
import pandas as pd
import altair as alt
import os
from altair_data_server import data_server
import seaborn as sns
import matplotlib.pyplot as plt
import vl_convert as vlc

# Save a vega-lite spec and a PNG blob for each plot in the notebook
alt.renderers.enable("mimetype")
alt.data_transformers.enable("data_server")
# Handle large data sets without embedding them in the notebook


# Reference: 531 Slack Channel by Joel
def save_chart(chart, filename, scale_factor=1):
    """
    Save an Altair chart using vl-convert

    Parameters
    ----------
    chart : altair.Chart
        Altair chart to save
    filename : str
        The path to save the chart to
    scale_factor: int or float
        The factor to scale the image resolution by.
        E.g. A value of `2` means two times the default resolution.
    """
    with alt.data_transformers.enable(
        "default"
    ) and alt.data_transformers.disable_max_rows():
        if filename.split(".")[-1] == "svg":
            with open(filename, "w") as f:
                f.write(vlc.vegalite_to_svg(chart.to_dict()))
        elif filename.split(".")[-1] == "png":
            with open(filename, "wb") as f:
                f.write(vlc.vegalite_to_png(chart.to_dict(), scale=scale_factor))
        else:
            raise ValueError("Only svg and png formats are supported")


def perform_eda(train_data_path, out_folder):
    if not (os.path.exists(out_folder)):
        os.makedirs(out_folder)

    train_df = pd.read_csv(
        train_data_path, index_col=0, encoding="utf-8"
    )
    
    categorical_features = [
        "neighbourhood_group",
        "room_type",
    ]

    numeric_features = [
        "price",
        "minimum_nights",
        "number_of_reviews",
        "calculated_host_listings_count",
        "availability_365",
        "days_from_last_review",
    ]

    # Plotting categorical features
    categorical_chart = (
        alt.Chart(train_df)
        .mark_bar()
        .encode(x=alt.X(alt.repeat(), type="nominal", sort="-y"), y=alt.Y("count()"))
        .properties(width=300, height=200)
        .repeat(categorical_features, columns=2)
    )

    save_chart(categorical_chart, out_path + "categorical_result.png")

    # Plotting numeric features.
    numeric_chart = (
        alt.Chart(train_df)
        .mark_bar()
        .encode(
            x=alt.X(alt.repeat(), type="quantitative", bin=alt.Bin(maxbins=30)),
            y=alt.Y("count()"),
        )
        .properties(width=300, height=200)
        .repeat(numeric_features, columns=2)
    )

    save_chart(numeric_chart, out_path + "numeric_result.png")

    # Plotting the geographical location of listings.
    geolocation_result = (
        alt.Chart(train_df)
        .mark_circle(size=6)
        .encode(
            y=alt.Y("latitude:Q", scale=alt.Scale(zero=False)),
            x=alt.X("longitude:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("neighbourhood_group:N"),
        )
    )

    save_chart(geolocation_result, out_path + "geolocation_result.png")

    # Plotting the geographical location of listings.
    lastreview_result = (
        alt.Chart(train_df, title="Distribution of days from last review")
        .mark_bar()
        .encode(
            x=alt.X(
                "days_from_last_review:N",
                title="Days from last review",
                bin=alt.Bin(maxbins=30),
            ),
            y=alt.Y("count()", title="Count of listings"),
        )
    )

    save_chart(lastreview_result, out_path + "lastreview_result.png")

    train_df.corr("spearman").to_csv(out_path + "corr.csv")

    train_df.describe().to_csv(out_path + "numeric_chart.csv")


if __name__ == "__main__":
    arguments = docopt(__doc__)

    train_data_path = arguments["--traindata"]  # Download 1 dataset at a time
    out_path = arguments["--output"][0]

    # Tests arguments
    assert train_data_path.endswith(".csv")
    assert "eda_results" in out_path

    perform_eda(train_data_path, out_path)

    # Tests that the files are generated as expected
    assert os.path.exists("./results/eda_results/categorical_result.png")
    assert os.path.exists("./results/eda_results/numeric_result.png")
    assert os.path.exists("./results/eda_results/geolocation_result.png")
    assert os.path.exists("./results/eda_results/lastreview_result.png")
    assert os.path.exists("./results/eda_results/corr.csv")
    assert os.path.exists("./results/eda_results/numeric_chart.csv")
