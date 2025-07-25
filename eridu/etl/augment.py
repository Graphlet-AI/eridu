import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession

from eridu.etl.acronyms import process_single_name

spark = SparkSession.builder.appName("Company Acronym Data Augmentation").getOrCreate()

abbrev_output_schema = T.ArrayType(
    T.StructType(
        [
            T.StructField("original", T.StringType(), True),
            T.StructField("abbreviated", T.StringType(), True),
        ]
    )
)


@F.pandas_udf(
    returnType="array<struct<original:string,abbreviated:string>>",
    functionType=F.PandasUDFType.SCALAR,
)
def generate_company_abbreviations(names: "pd.Series[str]") -> pd.Series:
    """
    Generate abbreviations for company names using cleanco to extract the basename, with and without periods.
    Returns an array of (original, abbreviation) pairs for each company.
    """

    # Apply to each name in the series
    return names.apply(process_single_name)
