import os
import json
import pathlib
from dotenv import load_dotenv
from openai import OpenAI
from extractor import extract

load_dotenv()

# Load registry
registry = {}
for path in pathlib.Path("schemas").glob("*.json"):
    with open(path) as f:
        schema = json.load(f)
        registry[schema["$id"]] = schema

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

text = ("This W-2 Wage and Tax Statement for the year 2016 reports the following information:"
        "Wages, tips, other compensation: $10,415.00 Federal Income tax withheld: $900.00"
        "Social security wages: $12,990.00 Social security tax withheld: $805.38"
        "Medicare wages and tips: $12,990.00 Medicare tax withheld: $188.36  Employee's SSA number: 123-45-6789"
        "Employer's FED ID number: 38-6005984  Control number: 00000001  Employer: Michigan State University,"
        "426 Auditorium #350, East Lansing MI 48824  Dependent care benefits: $4,500.00 (code DD in box 12a)"
        "Other (MN BSE): $600.00 (code E in box 12b)  Statutory Employee: Not checked  Retirement plan: Checked"
        "Third-Party Sick pay: Not checked  Employee: Sparty Jones, 123 Spartan Blvd, East Lansing MI 48823"
        "State: MI  State wages, tips, etc.: $10,415.00  Employer's state ID: 690350502  State income tax: $442.00")

result = extract(text, registry, llm=client)
print(json.dumps(result, indent=2))
