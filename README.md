# online-retail-data-analysis

  - Download the data from the following source and unzip the file:
    https://archive.ics.uci.edu/static/public/352/online+retail.zip

  - Convert the downloaded excel data to parquet format for further processing:
    ```
    df = pd.read_excel("Online Retail.xlsx", dtype={'InvoiceNo': str, 'StockCode': str, 'Description': str})
    df.to_parquet("Online Retail.parquet")
    ```

  - install the required libraries:
    ```
    $ python -mvenv exp_env
    $ source exp_env/bin/activate
    $ pip install pandas polars fireducks linetimer pycountry_convert  
    $ pip install --extra-index-url=https://pypi.nvidia.com "cudf-cu12==25.6.*" # For cuDF on cuda-12
    ```

  - Execute the sample demo programs:
    ```
    $ python src/demo_opt.py # For pandas
    $ python -mfireducks.pandas src/demo_opt.py # For FireDucks
    $ python -mcudf.pandas src/demo_opt.py # For cuDF
    $ python src/demo_opt_pl.py # For polars
    ```
