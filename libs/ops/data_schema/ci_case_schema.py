from schema import Schema, And, Use, Optional,Or

robustness_case_schema = Schema([
    {"id": Use(str),
    "env": Or('prod','pre-prod'),
    "notes":Use(str)}])